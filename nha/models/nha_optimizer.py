from nha.models.texture import MultiTexture
from nha.models.flame import *
from nha.models.offset_mlp import OffsetMLP
from nha.models.texture_mlp import NormalEncoder, TextureMLP
from nha.optimization.criterions import LeakyHingeLoss, MaskedCriterion
from nha.optimization.perceptual_loss import ResNetLOSS
from nha.optimization.holefilling_segmentation_loss import calc_holefilling_segmentation_loss
from nha.data.real import CLASS_IDCS, digitize_segmap
from nha.util.render import (
    normalize_image_points,
    batch_project,
    create_camera_objects,
    hard_feature_blend,
    render_shaded_mesh
)
from nha.util.general import (
    fill_tensor_background,
    seperated_gaussian_blur,
    masked_gaussian_blur,
    dict_2_device,
    DecayScheduler,
    erode_mask,
    softmask_gradient,
    IoU,
    NoSubmoduleWrapper,
    stack_dicts,
)
from nha.util.screen_grad import screen_grad
from nha.util.log import get_logger
from nha.util.lbs import batch_rodrigues

from pytorch3d.renderer import TexturesVertex, rasterize_meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import *

from pathlib import Path
from copy import copy
from argparse import ArgumentParser
from torchvision.transforms.functional import gaussian_blur
from torch.utils.data.dataloader import DataLoader

import shutil
import warnings
import torch
import torchvision
import pytorch_lightning as pl
import json
import numpy as np

logger = get_logger(__name__)


class NHAOptimizer(pl.LightningModule):
    """
    Main Class for Optimizing Neural Head Avatars from RGB sequences.
    """

    @staticmethod
    def add_argparse_args(parser):
        parser = ArgumentParser(parents=[parser], add_help=False)

        # specific arguments for combined module

        combi_args = [
            # texture settings
            dict(name_or_flags="--texture_hidden_feats", default=256, type=int),
            dict(name_or_flags="--texture_hidden_layers", default=8, type=int),
            dict(name_or_flags="--texture_d_hidden_dynamic", type=int, default=128),
            dict(name_or_flags="--texture_n_hidden_dynamic", type=int, default=1),
            dict(name_or_flags="--glob_rot_noise", type=float, default=5.0),
            dict(name_or_flags="--d_normal_encoding", type=int, default=32),
            dict(name_or_flags="--d_normal_encoding_hidden", type=int, default=128),
            dict(name_or_flags="--n_normal_encoding_hidden", type=int, default=2),
            dict(name_or_flags="--flame_noise", type=float, default=0.0),
            dict(name_or_flags="--soft_clip_sigma", type=float, default=-1.0),

            # geometry refinement mlp settings
            dict(name_or_flags="--offset_hidden_layers", default=8, type=int),
            dict(name_or_flags="--offset_hidden_feats", default=256, type=int),

            # FLAME settings
            dict(name_or_flags="--subdivide_mesh", type=int, default=1),
            dict(name_or_flags="--semantics_blur", default=3, type=int, required=False),
            dict(name_or_flags="--spatial_blur_sigma", type=float, default=0.01),

            # training timeline settings
            dict(name_or_flags="--epochs_offset", type=int, default=50,
                 help="Until which epoch to train flame parameters and offsets jointly"),
            dict(name_or_flags="--epochs_texture", type=int, default=500,
                 help="Until which epoch to train texture while keeping model fixed"),
            dict(name_or_flags="--epochs_joint", type=int, default=500,
                 help="Until which epoch to train model jointly while keeping model fixed"),
            dict(name_or_flags="--image_log_period", type=int, default=10),

            # lr settings
            dict(name_or_flags="--flame_lr", default=0.005, type=float, nargs=3),
            dict(name_or_flags="--offset_lr", default=0.005, type=float, nargs=3),
            dict(name_or_flags="--tex_lr", default=0.01, type=float, nargs=3),

            # loss weights
            dict(name_or_flags="--body_part_weights", type=str, required=True),
            dict(name_or_flags="--w_rgb", type=float, default=1, nargs=3),
            dict(name_or_flags="--w_perc", default=0, type=float, nargs=3),
            dict(name_or_flags="--w_lmk", default=1, type=float, nargs=3),
            dict(name_or_flags="--w_eye_closed", default=1, type=float, nargs=3),
            dict(name_or_flags="--w_edge", default=1, type=float, nargs=3),
            dict(name_or_flags="--w_norm", default=1, type=float, nargs=3),
            dict(name_or_flags="--w_lap", type=json.loads, nargs="*"),
            dict(name_or_flags="--w_shape_reg", default=1e-4, type=float, nargs=3),
            dict(name_or_flags="--w_expr_reg", default=1e-4, type=float, nargs=3),
            dict(name_or_flags="--w_pose_reg", default=1e-4, type=float, nargs=3),
            dict(name_or_flags="--w_surface_reg", default=1e-4, type=float, nargs=3),
            dict(name_or_flags="--texture_weight_decay", type=float, default=5e-6, nargs=3),
            dict(name_or_flags="--w_silh", type=json.loads, nargs="*"),
            dict(name_or_flags="--w_semantic_ear", default=[1e-2] * 3, type=float, nargs=3),
            dict(name_or_flags="--w_semantic_eye", default=[1e-1] * 3, type=float, nargs=3),
            dict(name_or_flags="--w_semantic_mouth", default=[1e-1] * 3, type=float, nargs=3),
            dict(name_or_flags="--w_semantic_hair", type=json.loads, nargs="*"),
        ]
        for f in combi_args:
            parser.add_argument(f.pop("name_or_flags"), **f)

        return parser

    def __init__(self, max_frame_id, w_lap, w_silh, w_semantic_hair, body_part_weights, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.callbacks = [pl.callbacks.ModelCheckpoint(filename="{epoch:02d}", save_last=True)]

        # flame model
        ignore_faces = np.load(FLAME_LOWER_NECK_FACES_PATH)  # ignore lower neck faces
        upsample_regions = dict(all=self.hparams["subdivide_mesh"])

        self._flame = FlameHead(
            FLAME_N_SHAPE,
            FLAME_N_EXPR,
            flame_template_mesh_path=FLAME_MESH_MOUTH_PATH,
            flame_model_path=FLAME_MODEL_PATH,
            flame_parts_path=FLAME_PARTS_MOUTH_PATH,
            ignore_faces=ignore_faces,
            upsample_regions=upsample_regions,
            spatial_blur_sigma=self.hparams["spatial_blur_sigma"],
        )

        self._shape = torch.nn.Parameter(torch.zeros(1, FLAME_N_SHAPE), requires_grad=True)
        self._expr = torch.nn.Parameter(torch.zeros(max_frame_id + 1, FLAME_N_EXPR), requires_grad=True)
        self._neck_pose = torch.nn.Parameter(torch.zeros(max_frame_id + 1, 3), requires_grad=True)
        self._jaw_pose = torch.nn.Parameter(torch.zeros(max_frame_id + 1, 3), requires_grad=True)
        self._eyes_pose = torch.nn.Parameter(torch.zeros(max_frame_id + 1, 6), requires_grad=True)
        self._translation = torch.nn.Parameter(torch.zeros(max_frame_id + 1, 3), requires_grad=True)
        self._rotation = torch.nn.Parameter(torch.zeros(max_frame_id + 1, 3), requires_grad=True)

        # restrict offsets to anything but eyeballs
        body_parts = self._flame.get_body_parts()
        eye_indices = np.concatenate([body_parts["left_eyeball"], body_parts["right_eyeball"]])
        indices = np.arange(len(self._flame.v_template))
        self._offset_indices = np.delete(indices, eye_indices)
        _, face_idcs = self._flame.faces_of_verts(torch.from_numpy(self._offset_indices), return_face_idcs=True)
        self._offset_face_indices = face_idcs

        self._vert_feats = torch.nn.Parameter(torch.zeros(1, len(self._offset_indices), 32), requires_grad=True)
        self._vert_feats.data.normal_(0.0, 0.02)

        cond_feats = 9
        in_feats = 3 + self._vert_feats.shape[-1]

        self._offset_mlp = OffsetMLP(
            in_feats=in_feats,
            cond_feats=cond_feats,
            hidden_feats=self.hparams["offset_hidden_feats"],
            hidden_layers=self.hparams["offset_hidden_layers"],
        )

        # appearance
        self._normal_encoder = NormalEncoder(
            input_dim=3,
            output_dim=self.hparams["d_normal_encoding"],
            hidden_layers_feature_size=self.hparams["d_normal_encoding_hidden"],
            hidden_layers=self.hparams["n_normal_encoding_hidden"],
        )

        teeth_res = 64
        face_res = 256
        d_feat = 64
        self._texture = TextureMLP(
            d_input=3 + d_feat,
            hidden_layers=self.hparams["texture_hidden_layers"],
            hidden_features=self.hparams["texture_hidden_feats"],
            d_dynamic=10 + 3,
            d_hidden_dynamic_head=self.hparams["texture_d_hidden_dynamic"],
            n_dynamic_head=self.hparams["texture_n_hidden_dynamic"],
            d_dynamic_head=self.hparams["d_normal_encoding"],
        )

        self._explFeatures = MultiTexture(
            torch.randn(d_feat, face_res, face_res) * 0.01,
            torch.randn(d_feat, teeth_res, teeth_res) * 0.01,
        )

        self._decays = {
            "lap": DecayScheduler(*w_lap, geometric=True),
            "semantic": DecayScheduler(*w_semantic_hair, geometric=False),
            "silh": DecayScheduler(*w_silh, geometric=False),
        }

        # body part weights
        with open(body_part_weights, "r") as f:
            key = "mlp"
            self._body_part_loss_weights = json.load(f)[key]
        self.semantic_labels = list(self._flame.get_body_parts().keys())

        # loss definitions
        self._leaky_hinge = LeakyHingeLoss(0.0, 1.0, 0.3)
        self._masked_L1 = MaskedCriterion(torch.nn.L1Loss(reduction="none"))

        if Path("assets/InsightFace/backbone.pth").exists():
            self._perceptual_loss = NoSubmoduleWrapper(ResNetLOSS())  # don't store perc_loss weights as model weights
        else:
            self._perceptual_loss = None

        # training stage
        self.fit_residuals = False
        self.is_train = False

        # learning rate for flame translation
        self._trans_lr = [0.1 * i for i in self.hparams["flame_lr"]]

        self._semantic_thr = 0.99
        self._blurred_vertex_labels = self._flame.spatially_blurred_vert_labels
        self._part_weights = None

        self._edge_mask = None

        # limits for pose and expression conditioning
        self.register_buffer("expr_min", torch.zeros(100) - 100)
        self.register_buffer("expr_max", torch.zeros(100) + 100)
        self.register_buffer("pose_min", torch.zeros(15) - 100)
        self.register_buffer("pose_max", torch.zeros(15) + 100)
        self.register_buffer("mouth_conditioning_min", torch.zeros(13) - 100)
        self.register_buffer("mouth_conditioning_max", torch.zeros(13) + 100)

    def on_train_start(self) -> None:

        # Copying config and body part weights to checkpoint dir
        logdir = Path(self.trainer.log_dir)
        conf_path = logdir / "config.ini"
        split_conf_path = logdir / "split_config.json"
        body_part_weights_path = logdir / "body_part_weights.json"

        if not conf_path.exists():
            shutil.copy(self.hparams["config"], conf_path, follow_symlinks=True)

        if not split_conf_path.exists():
            shutil.copy(self.hparams["split_config"], split_conf_path, follow_symlinks=True)

        if not body_part_weights_path.exists():
            shutil.copy(self.hparams["body_part_weights"], body_part_weights_path, follow_symlinks=True)

        # moves perceptual loss to own device
        try:
            self._perceptual_loss.to(self.device)
        except AttributeError:
            raise AttributeError("You have to download the backbone weights for the perceptual loss. Please refer"
                                 "to the 'Installation' section of the README")

        # hard setting lr
        flame_optim, offset_optim, tex_optim, joint_flame_optim, off_resid_optim, all_resid_optim = self.optimizers()

        lrs = self.get_current_lrs_n_lossweights()

        flame_optim.param_groups[0]["lr"] = lrs["flame_lr"]
        flame_optim.param_groups[1]["lr"] = lrs["trans_lr"]

        for pg in offset_optim.param_groups:
            pg["lr"] = lrs["offset_lr"]

        joint_flame_optim.param_groups[0]["lr"] = lrs["flame_lr"]
        joint_flame_optim.param_groups[1]["lr"] = lrs["trans_lr"]

        for pg in tex_optim.param_groups:
            pg["lr"] = lrs["tex_lr"]
            pg["weight_decay"] = lrs["texture_weight_decay"]

        off_resid_optim.param_groups[0]["lr"] = lrs["flame_lr"]
        off_resid_optim.param_groups[1]["lr"] = lrs["trans_lr"]

        all_resid_optim.param_groups[0]["lr"] = lrs["flame_lr"]
        all_resid_optim.param_groups[1]["lr"] = lrs["trans_lr"]

    def on_train_end(self) -> None:
        # determining dynamic condition extrema for validation
        self.get_dyn_cond_extrema(self.trainer.train_dataloader.dataset.datasets)

    def _get_current_optimizer(self, epoch=None):
        if epoch is None:
            epoch = self.current_epoch

        flame_optim, offset_optim, tex_optim, joint_flame_optim, off_resid_optim, all_resid_optim = self.optimizers()

        if self.fit_residuals:
            if epoch < self.hparams["epochs_offset"]:
                optim = [off_resid_optim]
            elif epoch < self.hparams["epochs_texture"] + self.hparams["epochs_offset"]:
                optim = None
            else:
                optim = [all_resid_optim]
        else:
            if epoch < self.hparams["epochs_offset"]:
                optim = [flame_optim, offset_optim]
            elif epoch < self.hparams["epochs_texture"] + self.hparams["epochs_offset"]:
                optim = [tex_optim]
            else:
                optim = [joint_flame_optim, offset_optim, tex_optim]
        return optim

    def _predict_offsets(self, neck_rot):
        """
        predicts vertex offsets, dynamic offsets are only enabled for neck region. Dynamic and static offsets are
        blended smoothly

        :param neck_rot: neck rotation vector of shape N x 3
        :return: offset tensor of shape N x V x 3
        """
        batch_size = len(neck_rot)
        final_offsets = torch.zeros(batch_size, len(self._flame.v_template), 3, device=self.device)
        v_temp = self._flame.v_template_normed
        v_temp = v_temp[self._offset_indices]

        neck = self._flame.apply_rotation_limits(neck=neck_rot)

        conditions = batch_rodrigues(neck.detach()).view(-1, 9)

        # increase batch_size by one for static mesh
        batch_size += 1

        # add zero condition to the end
        static_cond = torch.zeros_like(conditions[0])
        conditions = torch.cat([conditions, static_cond[None]], dim=0)

        # N x D -> N*V x D
        V = len(v_temp)

        # V x 3 -> N*V x 3
        v_temp = v_temp[None, ...]
        x = torch.cat([v_temp, self._vert_feats], dim=-1).expand(batch_size, -1, -1)

        mlp_out = self._offset_mlp(x, conditions)
        batch_size -= 1
        dynamic_offsets = mlp_out[:batch_size]
        static_offsets = mlp_out[[-1]].expand(batch_size, -1, -1)

        if self._blurred_vertex_labels.device != self.device:
            self._blurred_vertex_labels = self._blurred_vertex_labels.to(self.device)

        neck_index = self.semantic_labels.index("neck")
        alpha = self._blurred_vertex_labels[self._offset_indices, neck_index][None, :, None]

        min_alpha, max_alpha = 0.0, 1.0
        alpha = alpha.clamp(min_alpha, max_alpha)
        offsets = dynamic_offsets * alpha + (1 - alpha) * static_offsets

        # N * V x 3 -> N x V x 3
        final_offsets[:, self._offset_indices] = offsets.view(batch_size, V, 3)

        return final_offsets

    def _create_flame_param_batch(self, batch, ignore_shape=False, ignore_expr=False, ignore_pose=False,
                                  ignore_offsets=False):
        """
        adds residual para
        :param batch:
        :param ignore_shape:
        :param ignore_expr:
        :param ignore_pose:
        :param ignore_offsets:
        :return: dict with entries:
                    - shape: N x 300 FLAME shape parameters
                    - expr: N x 100 FLAME expression parameters
                    - neck: N x 3 neck rotation vector
                            (BEFORE APPLYING TANH TO ENFORCE ROTATION LIMITS, see flame._apply_rotation_limit())
                    - jaw: N x 3 jaw rotation vector
                            (BEFORE APPLYING TANH TO ENFORCE ROTATION LIMITS, see flame._apply_rotation_limit())
                    - eyes: N x 6 eye rotation vectors
                    - rotation: N x 3 global rotation vector
                    - translation: N x 3 global translation vector
                    - offsets: N x V x 3 vertex offset tensor
        """

        N = len(iter(batch.values()).__next__())
        indices = batch.get("frame", None)

        if indices is None:
            ignore_pose = True
            ignore_expr = True

        p = {}

        if ignore_shape:
            p["shape"] = 0
        else:
            p["shape"] = self._shape.expand(N, self._shape.shape[-1])

        if ignore_expr:
            p["expr"] = 0
        else:
            p["expr"] = self._expr[indices]

        if ignore_pose:
            p["neck"] = 0
            p["jaw"] = 0
            p["eyes"] = 0
            p["rotation"] = 0
            p["translation"] = 0
        else:
            p["neck"] = self._neck_pose[indices]  # * torch.tensor([[0, 1., 0]], device=self.device)
            p["jaw"] = self._jaw_pose[indices]
            p["eyes"] = self._eyes_pose[indices]
            p["rotation"] = self._rotation[indices]
            p["translation"] = self._translation[indices]

        # adding parameters from dataset
        p["shape"] = p["shape"] + batch.get("flame_shape", 0)
        p["expr"] = p["expr"] + batch.get("flame_expr", 0)
        p["translation"] = p["translation"] + batch.get("flame_trans", 0)
        p["rotation"] = p["rotation"] + batch["flame_pose"][:, :3]
        p["neck"] = p["neck"] + batch["flame_pose"][:, 3:6]
        p["jaw"] = p["jaw"] + batch["flame_pose"][:, 6:9]
        p["eyes"] = p["eyes"] + batch["flame_pose"][:, 9:15]

        if ignore_offsets:
            p["offsets"] = None
        else:
            p["offsets"] = self._predict_offsets(p["neck"])

        return p

    def _forward_flame(self, flame_params, return_mouth_conditioning=False):
        """
        Evaluates the flame head model

        :param flame_params: flame parameters dictionary as returned by self._create_flame_param_batch()
        :param return_mouth_conditioning: whether to return mouth conditioning for texture MLP or not
        :return:
            - vertices: N x V x 3
            - lmk: N x 68 x 3
            - (optional) mouth_conditioning: N x 13
        """

        flame_results = self._flame(
            **flame_params,
            zero_centered=True,
            use_rotation_limits=True,
            return_landmarks="static",
            return_mouth_conditioning=return_mouth_conditioning,
        )
        if len(flame_results) == 3:
            verts, lmks_static, mouth_conditioning = flame_results
        else:
            verts, lmks_static = flame_results
            mouth_conditioning = None

        if mouth_conditioning is None:
            return verts, lmks_static
        else:
            return verts, lmks_static, mouth_conditioning

    def _rasterize(self, meshes, cameras, image_size, center_prediction=False):
        """
        Rasterizes meshes using a standard rasterization approach
        :param meshes:
        :param cameras:
        :param image_size:
        :return: fragments:
                 screen_coords: N x H x W x 2  with x, y values following pytorch3ds NDC-coord system convention
                                top left = +1, +1 ; bottom_right = -1, -1
        """
        assert len(meshes) == len(cameras)

        eps = None
        verts_world = meshes.verts_padded()
        verts_view = cameras.get_world_to_view_transform().transform_points(verts_world, eps=eps)
        projection_trafo = cameras.get_projection_transform().compose(cameras.get_ndc_camera_transform())
        verts_ndc = projection_trafo.transform_points(verts_view, eps=eps)
        verts_ndc[..., 2] = verts_view[..., 2]

        if center_prediction:
            centers = 0.5 * (verts_ndc[..., :2].max(dim=-2, keepdim=True).values +
                             verts_ndc[..., :2].min(dim=-2, keepdim=True).values)
            verts_ndc[..., :2] = verts_ndc[..., :2] - centers

        meshes_ndc = meshes.update_padded(new_verts_padded=verts_ndc)

        verts_packed = meshes_ndc.verts_packed()
        faces_packed = meshes_ndc.faces_packed()
        face_verts = verts_packed[faces_packed]

        perspective_correct = cameras.is_perspective()
        znear = cameras.get_znear()
        if isinstance(znear, torch.Tensor):
            znear = znear.min().item()
        z_clip = None if not perspective_correct or znear is None else znear / 2

        with torch.no_grad():
            fragments = rasterize_meshes(
                meshes_ndc,
                image_size=image_size,
                blur_radius=0,
                faces_per_pixel=2,
                bin_size=None,
                max_faces_per_bin=None,
                clip_barycentric_coords=False,
                perspective_correct=True,
                cull_backfaces=False,
                z_clip_value=z_clip,
                cull_to_frustum=False,
            )

            vert_semantics = self._flame.vert_labels
            face_semantics = vert_semantics.repeat(len(meshes), 1)[meshes_ndc.faces_packed()]
            sem_labels = interpolate_face_attributes(fragments[0], fragments[2], face_semantics)

            # find region where eye overlaps eyeregion
            # left
            leye_idx = self.semantic_labels.index("left_eyeball")
            leyeregion_idx = self.semantic_labels.index("left_eye_region")
            left_overlaps = (sem_labels[..., [0], leye_idx] > 0) & (sem_labels[..., [1], leyeregion_idx] > 0)
            # right
            reye_idx = self.semantic_labels.index("right_eyeball")
            reyeregion_idx = self.semantic_labels.index("right_eye_region")
            right_overlaps = (sem_labels[..., [0], reye_idx] > 0) & (sem_labels[..., [1], reyeregion_idx] > 0)

            overlaps = (left_overlaps | right_overlaps).int()  # N x H x W x 1

            pix_to_face = fragments[0][..., [0]] * (1 - overlaps) + fragments[0][..., [1]] * overlaps
            zbuf = fragments[1][..., [0]] * (1 - overlaps) + fragments[1][..., [1]] * overlaps
            bary_coords = fragments[2][..., [0], :] * (1 - overlaps.unsqueeze(-1)) + \
                          fragments[2][..., [1], :] * overlaps.unsqueeze(-1)
            dists = fragments[3][..., [0]] * (1 - overlaps) + fragments[3][..., [1]] * overlaps

            fragments = Fragments(pix_to_face=pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists)

        # right now this only works with faces_per_pixel == 1
        pix2face, bary_coords = fragments.pix_to_face, fragments.bary_coords
        is_visible = pix2face[..., 0] > -1

        # shape (sum(is_visible), 3, 3)
        visible_faces = pix2face[is_visible][:, 0]
        visible_face_verts = face_verts[visible_faces]
        # shape (sum(is_visible), 3, 1)
        visible_bary_coords = bary_coords[is_visible][:, 0][..., None]

        visible_surface_point = visible_face_verts * visible_bary_coords
        visible_surface_point = visible_surface_point.sum(dim=1)

        screen_coords = torch.zeros(*pix2face.shape[:3], 2).to(self.device)
        screen_coords[is_visible] = visible_surface_point[:, :2]  # now have gradient

        # if images are not-squared we need to adjust the screen coordinates
        # by the aspect ratio => coords given as [-1,1] for shorter edge and
        # [-s,s] for longer edge where s is the aspect ratio
        H, W = image_size
        if H > W:
            s = H / W
            screen_coords[..., 1] *= 1 / s
        elif H < W:
            s = W / H
            screen_coords[..., 0] *= 1 / s

        return fragments, screen_coords

    def _rasterize_flame(self, batch, verts):
        """

        :param batch:
        :param verts:
        :return:
        """

        K = batch["cam_intrinsic"]
        RT = batch["cam_extrinsic"]

        if len(verts) != len(K):
            warnings.warn("Batch size and vertex batch size don't match. Cutting batch to have same length as verts.")
            K = K[: len(verts)]
            RT = RT[: len(verts)]

        H, W = batch["rgb"].shape[-2:]

        # FILL SCENE
        cameras = create_camera_objects(K, RT, (H, W), self.device)
        flame_meshes = Meshes(verts=verts, faces=self._flame.faces[None].expand(len(verts), -1, -1))

        # RASTERIZING FRAGMENTS
        return self._rasterize(flame_meshes, cameras, (H, W))

    def _render_silhouette(self, verts, K, RT, H, W, return_rasterizer_results=False, rasterized_results=None):
        """
        Render flame silhouette. ATTENTION: NO GRADIENT ATTACHED
        :param verts:
        :param K:
        :param RT:
        :param H:
        :param W:
        :return: tensor of shape N x 1 x H x W with values ranging from 0 ... 1
        """

        # FILL SCENE
        cameras = create_camera_objects(K, RT, (H, W), self.device)
        flame_meshes = Meshes(verts=verts, faces=self._flame.faces[None].expand(len(verts), -1, -1))

        # RASTERIZING FRAGMENTS
        if rasterized_results is not None:
            fragments, screen_coords = rasterized_results
        else:
            fragments, screen_coords = self._rasterize(flame_meshes, cameras, (H, W))
        N, H, W, K, _ = fragments.bary_coords.shape

        seg = (fragments.pix_to_face > -1).float().permute(0, 3, 1, 2)

        if return_rasterizer_results:
            return seg, dict(fragments=fragments, screen_coords=screen_coords)
        else:
            return seg

    def _render_normals(self, verts, K, RT, H, W, cameras=None, flame_meshes=None, faces=None,
                        return_rasterizer_results=False, rasterized_results=None):
        """
        renders tensor of normals of shape N x 3 x H x W
        :param batch:
        :param verts:
        :param faces: faces of shape N x F x 3, falls back to flame faces
        :return:
        """

        # FILL SCENE
        if cameras is None:
            cameras = create_camera_objects(K, RT, (H, W), self.device)
        if flame_meshes is None:
            faces = faces if faces is not None else self._flame.faces[None].expand(len(verts), -1, -1)
            flame_meshes = Meshes(verts=verts, faces=faces)

        # RASTERIZING FRAGMENTS
        if rasterized_results is not None:
            fragments, screen_coords = rasterized_results
        else:
            fragments, screen_coords = self._rasterize(flame_meshes, cameras, (H, W))
        N, H, W, K, _ = fragments.bary_coords.shape

        # NORMAL RENDERING
        face_normals = flame_meshes.verts_normals_padded()  # N x V_max x 3
        face_normals = cameras.get_world_to_view_transform().transform_normals(face_normals)
        face_normals = face_normals.flatten(end_dim=1)[flame_meshes.verts_padded_to_packed_idx()]  # N * V x 3
        face_normals = face_normals[flame_meshes.faces_packed()]
        pixel_face_normals = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_normals)

        blend_params = BlendParams(sigma=0, gamma=0, background_color=[0.0] * 3)
        normal_rendering = hard_feature_blend(pixel_face_normals, fragments, blend_params)
        normal_rendering = normal_rendering.permute(0, 3, 1, 2)  # shape N x 3+1 x H x W

        # flip orientations
        normal_rendering[:, :3] *= -1

        if return_rasterizer_results:
            return normal_rendering, dict(fragments=fragments, screen_coords=screen_coords)

        return normal_rendering   # shape N x 3+1 x H x W

    def _render_rgba(self, verts, K, RT, H, W, expr, pose, mouth_cond, return_rasterizer_results=False,
                     rasterized_results=None, rendered_normals=None, center_prediction=False):

        """
        render rgba image tensor of shape N x 4 x H x W
        first 3 channels normalized to -1 ... 1; 4th channel 0 ... 1

        if return_screen_coords == True: also returns dict with fragments and screen coords (N x H x W x 2) in ndc space

        :param verts:
        :param K:
        :param RT:
        :param H:
        :param W:
        :param expr: Nx 100
        :param: pose: N x 15
        :param: mouth_conditioning: N x 13
        :param rendered_normals: tensor of shape N x 3 x H x W of rendered and blended normals
                                 (serve as input for normal encoder)
        :return:
        """

        # FILL SCENE
        cameras = create_camera_objects(K, RT, (H, W), self.device)
        flame_meshes = Meshes(verts=verts, faces=self._flame.faces[None].expand(len(verts), -1, -1))

        # RASTERIZING FRAGMENTS
        if rasterized_results is not None and not center_prediction:
            fragments, screen_coords = rasterized_results
        else:
            fragments, screen_coords = self._rasterize(flame_meshes, cameras, (H, W),
                                                       center_prediction=center_prediction)
        N, H, W, faces_per_pix, _ = fragments.bary_coords.shape

        assert faces_per_pix == 1  # otherwise explicit texels might get messed up

        # FACE COORD RENDERING
        face_coords = self._flame.face_coords_normed.repeat(N, 1, 1)
        # shape: N x H x W x K x 3
        pixel_face_coords = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_coords)

        # EXPL TEXTURE SAMPLING
        mask = fragments.pix_to_face != -1
        uv_coords = self._flame.face_uvcoords.repeat(N, 1, 1)

        # shape: N x H x W x K x 2
        pixel_uv_coords = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, uv_coords)[..., :2]
        pixel_uv_ids = self._flame.face_uvmap.repeat(N)[fragments.pix_to_face]  # N x H x W x K
        pixel_uv_ids[~mask] = -1
        # pixel_expl_texels = expl_texels.permute(0, 2, 3, 1) \
        #     .unsqueeze(-2).expand(-1, -1, -1, K, -1)  # N x H x W x K x 3

        # predicting frequencies and phase shifts for different regions
        #  clip dynamic conditions
        expr, pose, mouth_cond = self._clip_expr_pose_mc(expr, pose, mouth_cond)

        mouth_frequencies, mouth_phase_shifts = self._texture.predict_frequencies_phase_shifts(mouth_cond)
        stat_frequencies, stat_phase_shifts = self._texture.predict_frequencies_phase_shifts(
            torch.zeros_like(mouth_cond))

        frequencies = torch.stack((mouth_frequencies, stat_frequencies), dim=1)  # N x 3 x C
        phase_shifts = torch.stack((mouth_phase_shifts, stat_phase_shifts), dim=1)  # N x 3 x C

        # rendering semantic map
        semantics = self._render_semantics(verts, K, RT, H, W, rasterized_results=[fragments, screen_coords],
                                           return_confidence=True).detach()

        with torch.no_grad():
            mouth_weights = semantics[:, self.semantic_labels.index("mouth")]
            face_weights = (semantics[:, -1] - mouth_weights).clip(min=0)
            region_weights = torch.stack((mouth_weights, face_weights), dim=-1)  # N x H x W x 3
            region_weights = region_weights.view(N, H, W, 1, -1)  # N x H x W x K x 3

        if rendered_normals is None:
            rendered_normals = self._render_normals(verts, K, RT, H, W, cameras=cameras, flame_meshes=flame_meshes,
                                                    rasterized_results=[fragments, screen_coords], )[:, :3]

        # MLP TEXTURE SAMPLING
        pixel_face_coords_masked = pixel_face_coords[mask]
        pixel_uv_coords_masked = pixel_uv_coords[mask]
        pixel_uv_ids_masked = pixel_uv_ids[mask]

        if getattr(self, "n_upsample", 1) != 1:
            rendered_normals = torchvision.transforms.functional.resize(rendered_normals, (
                    torch.tensor(rendered_normals.shape[-2:]) / self.n_upsample).int().tolist())
        normal_encoding = self._normal_encoder(rendered_normals)  # N x C x H x W

        if getattr(self, "n_upsample", 1) != 1:
            normal_encoding = torchvision.transforms.functional.resize(normal_encoding, (H, W))

        pixel_normal_encoding = normal_encoding.permute(0, 2, 3, 1)
        pixel_normal_encoding = pixel_normal_encoding.unsqueeze(-2).expand(-1, -1, -1, faces_per_pix,
                                                                           -1)  # N x H x W x K x D
        pixel_normal_encoding_masked = pixel_normal_encoding[mask]
        pixel_expl_features_masked = self._explFeatures(pixel_uv_coords_masked.view(1, 1, -1, 2),
                                                        pixel_uv_ids_masked.view(1, 1, -1))
        pixel_expl_features_masked = pixel_expl_features_masked[0, :, 0, :].permute(1, 0)
        static_mlp_conditions_masked = torch.cat((pixel_face_coords_masked, pixel_expl_features_masked), dim=-1)
        n_masked = torch.arange(N).view(N, 1, 1, 1).expand(N, H, W, faces_per_pix)[mask]
        region_weights_masked = region_weights[mask]  # N' x 3
        frequencies_masked = torch.sum(frequencies[n_masked] * region_weights_masked.unsqueeze(-1), dim=1)
        phase_shifts_masked = torch.sum(phase_shifts[n_masked] * region_weights_masked.unsqueeze(-1), dim=1)
        # N' x C

        mlp_pixel_rgb = torch.zeros(N, H, W, faces_per_pix, 3, dtype=fragments.bary_coords.dtype,
                                    device=fragments.bary_coords.device)

        mlp_pixel_rgb[mask] = self._texture.forward(static_mlp_conditions_masked, frequencies_masked,
                                                    phase_shifts_masked,
                                                    dynamic_conditions=pixel_normal_encoding_masked)

        blend_params = BlendParams(sigma=0, gamma=0, background_color=[1.0] * 3)
        mlp_rgba_pred = hard_feature_blend(mlp_pixel_rgb, fragments, blend_params)
        # N x H x W x 3 + 1

        # if random_select:
        #     feature_img[~mask.expand(-1, -1, -1, 4)] = 1

        mlp_rgba_pred = mlp_rgba_pred.permute(0, 3, 1, 2)

        if return_rasterizer_results:
            return mlp_rgba_pred, dict(fragments=fragments, screen_coords=screen_coords)
        else:
            return mlp_rgba_pred  # N x 4 x H x W

    @torch.no_grad()
    def get_dyn_cond_extrema(self, dataset=None):
        dataset = dataset if dataset is not None else self.trainer.train_dataloader.dataset.datasets

        dataloader = DataLoader(dataset, batch_size=48, num_workers=4)

        for i, batch in enumerate(dataloader):
            batch = dict_2_device(batch, self.device)
            flame_params_offsets = self._create_flame_param_batch(batch)
            _, _, mc = self._forward_flame(flame_params_offsets, return_mouth_conditioning=True)
            expr = flame_params_offsets["expr"]
            pose = torch.cat((flame_params_offsets["rotation"], flame_params_offsets["neck"],
                              flame_params_offsets["jaw"], flame_params_offsets["eyes"]), dim=1)

            if i == 0:
                self.expr_min = torch.min(expr, dim=0).values
                self.expr_max = torch.max(expr, dim=0).values
                self.pose_min = torch.min(pose, dim=0).values
                self.pose_max = torch.max(pose, dim=0).values
                self.mouth_conditioning_min = torch.min(mc, dim=0).values
                self.mouth_conditioning_max = torch.max(mc, dim=0).values
            else:
                self.expr_min = torch.min(self.expr_min, torch.min(expr, dim=0).values)
                self.expr_max = torch.max(self.expr_max, torch.max(expr, dim=0).values)
                self.pose_min = torch.min(self.pose_min, torch.min(pose, dim=0).values)
                self.pose_max = torch.max(self.pose_max, torch.max(pose, dim=0).values)
                self.mouth_conditioning_min = torch.min(self.mouth_conditioning_min, torch.min(mc, dim=0).values)
                self.mouth_conditioning_max = torch.max(self.mouth_conditioning_max, torch.max(mc, dim=0).values)

        return (
            (self.expr_min, self.expr_max),
            (self.pose_min, self.pose_max),
            (self.mouth_conditioning_min, self.mouth_conditioning_max),
        )

    def _clip_expr_pose_mc(self, expr, pose, mouth_conditioning):
        if not self.training:
            expr = expr.clip(min=self.expr_min, max=self.expr_max)
            pose = pose.clip(min=self.pose_min, max=self.pose_max)
            mouth_conditioning = mouth_conditioning.clip(min=self.mouth_conditioning_min,
                                                         max=self.mouth_conditioning_max)

        #  Soft Clipping: conditions are scaled such that "normal" movements result in parameter changes of roughly
        #  +-2 before applying tanh. sigma can be set as hyperparameter to strengthen (high sigma) or weaken (low sigma)
        #  non-linear component of clipping
        sig = self.hparams["soft_clip_sigma"]
        if sig > 0:
            expr = torch.tanh(expr * sig)
            pose = torch.tanh(
                torch.cat((pose[..., :3] / (np.pi / 4.0),  # global rot in angles with native range +- pi/2
                           pose[..., 3:9],
                           pose[..., 9:] / (np.pi / 4.0),  # eye rot in angles with native range +- pi/2
                           ), dim=-1, ) * sig)
            mouth_conditioning = torch.cat(
                (
                    mouth_conditioning[..., :-3],
                    torch.tanh(mouth_conditioning[..., -3:] / (np.pi / 4.0) * sig),
                ),
                dim=-1,
            )

        return expr, pose, mouth_conditioning

    def _render_semantics(self, verts, K, RT, H, W, return_rasterizer_results=False, rasterized_results=None,
                          return_confidence=False):
        """
        render semantic image tensor of shape N x C+1 x H x W (last channel is bg channel)
        entries between 0 ... 1

        if return_screen_coords == True: also returns dict with fragments and screen coords (N x H x W x 2) in ndc space

        :param verts:
        :param K:
        :param RT:
        :param H:
        :param W:
        :return:
        """
        # FILL SCENE
        cameras = create_camera_objects(K, RT, (H, W), self.device)
        flame_meshes = Meshes(verts=verts, faces=self._flame.faces[None].expand(len(verts), -1, -1))

        # RASTERIZING FRAGMENTS
        if rasterized_results is not None:
            fragments, screen_coords = rasterized_results
        else:
            fragments, screen_coords = self._rasterize(flame_meshes, cameras, (H, W))
        N, H, W, K, _ = fragments.bary_coords.shape

        # get semantics labels and optionally confidence
        face_semantics = self._flame.vert_labels
        if return_confidence:
            face_semantics = torch.cat([face_semantics, self._blurred_vertex_labels], dim=1)

        C = face_semantics.shape[-1]
        face_semantics = face_semantics.repeat(N, 1)[flame_meshes.faces_packed()]
        pixel_face_semantics = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords,
                                                           face_semantics)  # N x H x W x K x C

        blend_params = BlendParams(sigma=0, gamma=0, background_color=[0.0] * C)
        semantic_rendering = hard_feature_blend(pixel_face_semantics, fragments, blend_params)  # N x H x W x C+1
        semantic_rendering = semantic_rendering.permute(0, 3, 1, 2)  # N x C+1 x H x W

        if return_rasterizer_results:
            return semantic_rendering, dict(fragments=fragments, screen_coords=screen_coords)
        else:
            return semantic_rendering  # N x C+1 x H x W

    #
    # LOSS TERMS START HERE
    #
    def _blur_vertex_labels(self, vert_labels, blur_iters=2):

        # 1. we build the filter matrix
        meshes = Meshes(verts=[self._flame.v_template], faces=[self._flame.faces])
        verts = meshes.verts_packed()
        edges = meshes.edges_packed()

        V = verts.shape[0]
        e0, e1 = edges.unbind(1)

        # all neighbors
        idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
        idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)

        # and vertices themselves
        diag_0 = torch.arange(V, device=verts.device)
        diag = torch.stack([diag_0, diag_0], dim=0).T
        idx = torch.cat([idx01, idx10, diag], dim=0).t()

        # First, we construct the adjacency matrix including diagonal,
        # i.e. A[i, j] = 1 if (i,j) is an edge, or
        # A[e0, e1] = 1 &  A[e1, e0] = 1
        ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
        A = torch.sparse.FloatTensor(idx, ones, (V, V))

        # the sum of i-th row of A gives the degree of the i-th vertex
        deg = torch.sparse.sum(A, dim=1).to_dense()

        # We construct the filter matrix by dividing by the degree
        deg0 = 1 / deg[e0]
        deg1 = 1 / deg[e1]
        deg_diag = 1 / deg[diag_0]
        val = torch.cat([deg0, deg1, deg_diag])
        deg_mat = torch.sparse.FloatTensor(idx, val, (V, V))

        for i in range(blur_iters):
            vert_labels = deg_mat.mm(vert_labels)

        return vert_labels

    def _compute_laplacian_smoothing_loss(self, vertices, offset_vertices):
        batch_faces = self._flame.faces[None, ...]
        batch_faces = batch_faces[:, self._offset_face_indices]
        batch_faces = batch_faces.expand(vertices.shape[0], *batch_faces.shape[1:])

        basis_meshes = Meshes(verts=vertices, faces=batch_faces)
        offset_meshes = Meshes(verts=offset_vertices, faces=batch_faces)

        # compute weights
        N = len(offset_meshes)
        basis_verts_packed = basis_meshes.verts_packed()  # (sum(V_n), 3)
        offset_verts_packed = offset_meshes.verts_packed()

        with torch.no_grad():
            L = basis_meshes.laplacian_packed()
            basis_lap = L.mm(basis_verts_packed)  # .norm(dim=1)  * weights

        offset_lap = L.mm(offset_verts_packed)  # .norm(dim=1)  * weights
        diff = (offset_lap - basis_lap) ** 2
        diff = diff.sum(dim=1)
        diff = diff.view(N, -1)

        # re-weight by body parts
        if self._part_weights is None:
            parts = self._flame.get_body_parts()
            weights = self._body_part_loss_weights["lap"]
            part_weights = torch.ones(vertices.shape[1], 1, device=self.device)
            for part, weight in weights.items():
                if part == "teeth":
                    continue
                if "ear" in part:
                    continue
                part_weights[parts[part]] *= weight
            part_weights = self._blur_vertex_labels(part_weights, 3)
            for part, weight in weights.items():
                if "ear" in part:
                    part_weights[parts[part]] = weight
            self._part_weights = part_weights

        diff *= self._part_weights[:, 0].view(1, -1)
        return diff.sum() / N

    def _compute_silhouette_loss(self, batch, verts, rasterized_results=None, return_iou=False):
        gt_mask = batch["seg"]
        K = batch["cam_intrinsic"]
        RT = batch["cam_extrinsic"]
        N, V, _ = verts.shape
        H, W = gt_mask.shape[-2:]

        semantics_pred, raster_res = self._render_semantics(verts, K=K, RT=RT, H=H, W=W, return_rasterizer_results=True,
                                                            rasterized_results=rasterized_results)

        pred_mask = semantics_pred[:, [-1]]

        screen_coords = raster_res["screen_coords"] * (-1)  # N x H_r x W_r x 2

        loss, iou = calc_holefilling_segmentation_loss(gt_mask.float(), pred_mask.float(), screen_coords,
                                                       return_iou=True, sigma=2.0)

        if return_iou:
            return loss, iou.mean()
        else:
            return loss

    def _compute_semantic_loss(self, batch, vertices_offsets, rasterized_results=None, return_total_iou=False):
        """
        calculates semantic loss for ears, eyes, mouth, and hair
        :param batch:
        :param vertices_offsets:
        :param rasterized_results:
        :return:
        """
        parsing_gt = batch["parsing"]
        K = batch["cam_intrinsic"]
        RT = batch["cam_extrinsic"]
        H, W = batch["rgb"].shape[-2:]

        # predict semantics and extract results
        prediction, raster_res = self._render_semantics(
            vertices_offsets,
            K,
            RT,
            H,
            W,
            return_rasterizer_results=True,
            rasterized_results=rasterized_results,
            return_confidence=True,
        )
        screen_coords = raster_res["screen_coords"] * (-1)  # N x H_r x W_r x 2
        semantics_pred = prediction[:, : len(self.semantic_labels)]
        confidence = prediction[:, len(self.semantic_labels): -1]
        alpha = prediction[:, [-1]]

        # EAR SEGMENTATION
        l_ear_index = self.semantic_labels.index("left_ear")
        r_ear_index = self.semantic_labels.index("right_ear")

        pred_ear = semantics_pred[:, [l_ear_index]] > self._semantic_thr
        pred_ear |= semantics_pred[:, [r_ear_index]] > self._semantic_thr
        confidence_ear = confidence[:, [l_ear_index]] + confidence[:, [r_ear_index]]
        gt_ear = (parsing_gt == CLASS_IDCS["l_ear"]) | (parsing_gt == CLASS_IDCS["r_ear"])

        loss_ear, iou_ear = calc_holefilling_segmentation_loss(
            gt_ear.float(),
            pred_ear.float(),
            screen_coords,
            return_iou=True,
            sigma=2.0,
            grad_weight=confidence_ear ** 10,
        )

        # EYE + MOUTH SEGMENTATION
        trust_scores = batch["trust_semantics"]
        # if torch.sum(valid_eyes) == 0:
        #     loss_eye, iou_eye = 0, 1
        # else:
        # only do eye segmentation if gt labels can be trusted
        l_eye_index = self.semantic_labels.index("left_eyeball")
        r_eye_index = self.semantic_labels.index("right_eyeball")

        pred_eye = semantics_pred[:, [l_eye_index]] < self._semantic_thr
        pred_eye &= semantics_pred[:, [r_eye_index]] < self._semantic_thr
        gt_eye = (parsing_gt != CLASS_IDCS["l_eye"]) & (parsing_gt != CLASS_IDCS["r_eye"])

        loss_eye, iou_eye = calc_holefilling_segmentation_loss(
            gt_eye.float(),
            pred_eye.float(),
            screen_coords,
            return_iou=True,
            sigma=1.0,
            sample_weights=trust_scores,
        )

        # MOUTH SEGMENTATION
        # only do mouth segmentation if gt labels can be trusted
        mouth_index = self.semantic_labels.index("mouth")
        pred_mouth = semantics_pred[:, [mouth_index]] < self._semantic_thr
        gt_mouth = parsing_gt != CLASS_IDCS["mouth"]
        # gt_mouth &= ( parsing_gt != CLASS_IDCS["teeth"])

        loss_mouth, iou_mouth = calc_holefilling_segmentation_loss(
            gt_mouth.float(),
            pred_mouth.float(),
            screen_coords,
            return_iou=True,
            sigma=1.0,
            sample_weights=trust_scores,
        )

        # Hair SEGMENTATION
        # prediction
        scalp_pred = semantics_pred[:, [self.semantic_labels.index("scalp")]] > self._semantic_thr
        neck_pred = semantics_pred[:, [self.semantic_labels.index("neck")]] > self._semantic_thr
        hair_pred = scalp_pred & ~neck_pred  # hair is scalp without neck
        skin_pred = (alpha > 0) & ~hair_pred  # skin is fg without hair
        pred_head = torch.cat([hair_pred, skin_pred], dim=1).float()

        # confidence
        scalp_conf = confidence[:, [self.semantic_labels.index("scalp")]]
        neck_conf = confidence[:, [self.semantic_labels.index("neck")]]
        hair_conf = scalp_conf * (1 - neck_conf)
        skin_conf = alpha * (1 - hair_conf)
        confidence_head = torch.cat([hair_conf, skin_conf], dim=1)

        # gt
        hair_gt = parsing_gt == CLASS_IDCS["hair"]
        cloth_gt = parsing_gt == CLASS_IDCS["cloth"]
        bg_gt = parsing_gt == CLASS_IDCS["background"]
        skin_gt = ~bg_gt & ~cloth_gt & ~hair_gt
        gt_head = torch.cat([hair_gt, skin_gt], dim=1).float()

        loss_hair, iou_hair = calc_holefilling_segmentation_loss(
            gt_head,
            pred_head,
            screen_coords,
            return_iou=True,
            sigma=2.0,
            grad_weight=confidence_head ** 10,
        )

        retval = [
            [loss_ear, loss_eye, loss_mouth, loss_hair],
            [iou_ear, iou_eye, iou_mouth, iou_hair],
            semantics_pred,
        ]
        if return_total_iou:
            retval.append(IoU(alpha > 0, ~bg_gt & ~cloth_gt))
        return retval

    def _compute_rgb_losses(self, batch, verts, expr, pose, mouth_conditioning, rasterized_results=None,
                            rendered_normals=None):
        """
        Computes photometric and perceptual loss
        :returns dict("rgb_loss"=scalar_rgb_loss_tensor, "perc_loss"=scalar_perc_loss_tensor)
        """

        rgb_gt = batch["rgb"]
        mask = batch["seg"].detach()

        K = batch["cam_intrinsic"]
        RT = batch["cam_extrinsic"]
        H, W = rgb_gt.shape[-2:]

        rgba_pred, raster_dict = self._render_rgba(verts, K, RT, H, W, expr=expr, pose=pose,
                                                   mouth_cond=mouth_conditioning, rendered_normals=rendered_normals,
                                                   return_rasterizer_results=True,
                                                   rasterized_results=rasterized_results)

        predicted_images = rgba_pred[:, :3]
        predicted_seg = rgba_pred[:, [3]].detach()

        screen_coords = raster_dict["screen_coords"] * (-1)

        screen_colors = screen_grad(batch["rgb"], screen_coords)

        # use intersection mask
        mask = predicted_seg * mask
        rgb_loss = self._masked_L1(predicted_images, screen_colors, mask)

        # ATTENTION: perceptual loss only provides gradient to texture!
        # don't apply gradient to boundary of silhouette -> artifacts occur otherwise
        gradient_margin = int(max(H, W) / 50)
        predicted_images = softmask_gradient(predicted_images, erode_mask(predicted_seg, gradient_margin))
        perc_loss = self._perceptual_loss(predicted_images, screen_colors.detach()).mean() if \
            self.get_current_lrs_n_lossweights()["w_perc"] >= 0 else 0.0

        return dict(rgb_loss=rgb_loss, perc_loss=perc_loss)

    def _compute_lmk_loss(self, batch, pred_lmks):
        lmks = batch["lmk2d"].clone()
        # 2d landmarks origin top-left, first coordinate x, second y
        lmks, confidence = lmks[:, :, :2], lmks[:, :, 2]
        K = batch["cam_intrinsic"]
        RT = batch["cam_extrinsic"]
        rgb = batch["rgb"]

        img_size = rgb.shape[-2:]

        # ignore eye mid lmks
        confidence[:, [37, 38, 43, 44, 41, 40, 47, 46]] = 0

        # normalize original landmarks to image resolution
        lmks[:, :, 0], lmks[:, :, 1] = normalize_image_points(lmks[:, :, 0], lmks[:, :, 1], img_size)
        proj_pred_lmks = batch_project(pred_lmks, K, RT, img_size, self.device, normalize=True)
        proj_pred_lmks, z = proj_pred_lmks[:, :, :2], proj_pred_lmks[:, :, 2]

        lmk_loss = torch.norm(lmks - proj_pred_lmks, dim=2, p=1) * confidence
        return lmk_loss[:, :68].mean()

    def _compute_eye_closed_loss(self, flame_params, batch):

        # left eye:  [38,42], [39,41] - 1
        # right eye: [44,48], [45,47] -1
        K = batch["cam_intrinsic"]
        RT = batch["cam_extrinsic"]
        rgb = batch["rgb"]
        img_size = rgb.shape[-2:]

        for k in flame_params.keys():
            if k != "expr" and flame_params[k] is not None:
                flame_params[k] = flame_params[k].detach()

        _, flame_lmks = self._forward_flame(flame_params)

        proj_pred_lmks = batch_project(flame_lmks, K, RT, img_size, self.device, normalize=False)
        flame_lmks, _ = proj_pred_lmks[:, :, :2], proj_pred_lmks[:, :, 2]
        eye_up = flame_lmks[:, [37, 38, 43, 44], :] / img_size[0]
        eye_bottom = flame_lmks[:, [41, 40, 47, 46], :] / img_size[0]
        pred_dist = torch.sqrt(((eye_up - eye_bottom) ** 2).sum(2) + 1e-7)  # [bz, 4]

        left_dist, right_dist = batch["eye_distance"]
        gt_dists = torch.stack([right_dist, right_dist, left_dist, left_dist], dim=1) / img_size[0]

        # compare only when semantics are trusted
        # pred_dist = pred_dist[batch['trust_semantics']]
        # gt_dists = gt_dists[batch['trust_semantics']]
        diff = (pred_dist - gt_dists).abs() * batch["trust_semantics"].view(-1, 1)
        return diff.sum() / len(flame_lmks)

    def _compute_flame_reg_losses(self, batch):
        indices = torch.unique(batch["frame"])
        shape_reg = torch.sum(self._shape ** 2) / 2
        expr_reg = torch.sum(self._expr[indices] ** 2) / 2
        pose = torch.cat([self._neck_pose[indices], self._jaw_pose[indices], self._eyes_pose[indices]], dim=1)
        pose_reg = torch.sum(pose ** 2) / 2
        return shape_reg, expr_reg, pose_reg

    def _get_normal_mask(self, batch):
        """
        Creates normal mask by combining groundtruth normal mask and face parsing mask
        :param batch:
        :return: normal mask
        """

        N, _, H, W = batch["rgb"].shape
        parsing = batch["parsing"]
        relevant_idcs = [
            CLASS_IDCS["skin"],
            CLASS_IDCS["nose"],
            # CLASS_IDCS["l_eye"],
            # CLASS_IDCS["r_eye"],
            CLASS_IDCS["l_brow"],
            CLASS_IDCS["r_brow"],
            CLASS_IDCS["u_lip"],
            CLASS_IDCS["l_lip"],
        ]

        gt_normals = batch["normal"]
        gt_normal_mask = torch.any((gt_normals.abs() > 0.004), dim=1).float().unsqueeze(1)
        gt_normal_mask = erode_mask(gt_normal_mask, int(max(H, W) / 50))

        mask = torch.zeros(N, 1, H, W, device=self.device).bool()
        for idx in relevant_idcs:
            mask = mask | (parsing == idx)
        return mask.float() * gt_normal_mask

    def _compute_normal_loss(self, batch, vertices_offsets, rendered_semantics, rasterized_results=None,
                             return_normal_pred=False):
        """

        :param batch:
        :param vertices_offsets:
        :param rendered_semantics: tensor of shape N x C x H x W with rendered semantics of mesh
        :param rasterized_results:
        :return:
        """

        normal_gt = batch["normal"]
        N, _3, H, W = normal_gt.shape
        # normal_gt[:, :2] = -1 * normal_gt[:, :2]  # mirroring gt normals

        mask = self._get_normal_mask(batch)
        K = batch["cam_intrinsic"]
        RT = batch["cam_extrinsic"]

        normal_pred, raster_res = self._render_normals(vertices_offsets, K, RT, H, W, return_rasterizer_results=True,
                                                       rasterized_results=rasterized_results)

        normal_pred, seg_pred = normal_pred[:, :3], normal_pred[:, [3]]

        # compute the intersection with the rendered mask
        mask = mask * seg_pred

        # excluding specific mesh regions
        mask = mask * (rendered_semantics[:, [self.semantic_labels.index("mouth")]] < 0.1).float()
        mask *= (rendered_semantics[:, [self.semantic_labels.index("left_eyeball")]] < 0.1).float()
        mask *= (rendered_semantics[:, [self.semantic_labels.index("right_eyeball")]] < 0.1).float()

        valid_eyes = batch["trust_semantics"] > 0.5
        if torch.sum(~valid_eyes) > 0:
            l_eyeregion = rendered_semantics[:, [self.semantic_labels.index("left_eye_region")]]
            mask[~valid_eyes] *= (l_eyeregion < 0.1).float()[~valid_eyes]
            r_eyeregion = rendered_semantics[:, [self.semantic_labels.index("right_eye_region")]]
            mask[~valid_eyes] *= (r_eyeregion < 0.1).float()[~valid_eyes]

        # calculating normal laplacians
        sigma = max(H, W) / 50
        with torch.no_grad():
            blurred_normal_gt = masked_gaussian_blur(normal_gt.detach(), mask, sigma=sigma, kernel_size=None,
                                                     blur_fc=seperated_gaussian_blur)
            blurred_normal_pred = masked_gaussian_blur(normal_pred.detach(), mask, sigma=sigma, kernel_size=None,
                                                       blur_fc=seperated_gaussian_blur)

        normal_gt_lapl = normal_gt - blurred_normal_gt
        normal_pred_lapl = normal_pred - blurred_normal_pred

        # shape [N, H, W, 2]
        screen_coords = raster_res["screen_coords"] * (-1)

        normal_gt_lapl = screen_grad(normal_gt_lapl, screen_coords)

        loss = self._masked_L1(normal_gt_lapl, normal_pred_lapl, mask=mask)

        if return_normal_pred:
            return loss, normal_pred
        else:
            return loss
        # return self._masked_L1(normal_gt, normal_pred, mask=mask)

    def _compute_surface_consistency(self, offsets):

        N = len(offsets)

        if N == 1:
            return 0

        # offsets = offsets.view(N, -1)

        # for each frame select another frame and compare their offsets
        all_indices = list(range(N))
        possible_partners = [all_indices[:i] + all_indices[i + 1:] for i in all_indices]
        partner_indices = [np.random.choice(p) for p in possible_partners]

        frame_i_offsets = offsets
        frame_j_offsets = offsets[partner_indices]

        diff = torch.abs(frame_i_offsets - frame_j_offsets.detach())

        return diff.sum() / N

    def _compute_edge_length_loss(self, offset_vertices):

        if self._edge_mask is None:
            # compute edge mask that later limits the calculation to scalp vertices only
            meshes = Meshes(verts=[self._flame.v_template], faces=[self._flame.faces])
            edges_packed = meshes.edges_packed()
            bp = self._flame.get_body_parts()
            mask = torch.zeros_like(edges_packed).bool()
            for i in bp["scalp"]:
                mask |= edges_packed == i

            self._edge_mask = mask.sum(dim=1) > 0

        batch_faces = self._flame.faces[None, ...]
        batch_faces = batch_faces.expand(offset_vertices.shape[0], *batch_faces.shape[1:])
        meshes = Meshes(verts=offset_vertices, faces=batch_faces)

        N = len(meshes)
        edges_packed = meshes.edges_packed()  # (sum(E_n), 2)
        verts_packed = meshes.verts_packed()  # (sum(V_n), 3)

        verts_edges = verts_packed[edges_packed]
        v0, v1 = verts_edges.unbind(1)
        lengths = (v0 - v1).norm(dim=1, p=2)

        # add batch dimension and filter scalp edges
        lengths = lengths.view(N, -1)[:, self._edge_mask]
        mean_lengths = torch.mean(lengths, dim=1, keepdim=True).expand(lengths.shape).detach()
        deviation = torch.abs(lengths - mean_lengths)

        # calculate loss only when larger than 3x mean
        deviation[deviation < 1.5 * mean_lengths] *= 0
        return deviation.mean()

    def _optimize_offsets(self, batch):
        """
        Optimize the Flame head model + offets based on silhouette, normal map and semantic map

        :param batch: input data batch as returned from self.prepare_batch()
        :return: scalar loss, log dict
        """

        # get loss term weights
        w_lap = self._decays["lap"].get(self.current_epoch)
        w_semantic_hair = self._decays["semantic"].get(self.current_epoch)
        w_silh = self._decays["silh"].get(self.current_epoch)

        # computes losses
        flame_params = self._create_flame_param_batch(batch, ignore_offsets=True)
        vertices, _ = self._forward_flame(flame_params)

        flame_params_offsets = self._create_flame_param_batch(batch)
        offsets_verts, pred_lmks = self._forward_flame(flame_params_offsets)

        # precompute rasterization results to safe computation time
        raster_res = self._rasterize_flame(batch, offsets_verts)

        silh_loss = self._compute_silhouette_loss(batch, offsets_verts, rasterized_results=raster_res)

        sem_res = self._compute_semantic_loss(batch, offsets_verts, rasterized_results=raster_res,
                                              return_total_iou=True)
        semantic_loss, semantic_iou, semantic_pred, total_iou = sem_res
        sem_ear, sem_eye, sem_mouth, sem_hair = semantic_loss
        iou_ear, iou_eye, iou_mouth, iou_hair = semantic_iou

        lap_loss = self._compute_laplacian_smoothing_loss(vertices, offsets_verts)

        normal_loss = self._compute_normal_loss(
            batch,
            offsets_verts,
            rendered_semantics=semantic_pred,
            rasterized_results=raster_res,
        )

        lmk_loss = self._compute_lmk_loss(batch, pred_lmks)

        eye_closed_loss = self._compute_eye_closed_loss(copy(flame_params), batch)

        surface_reg = self._compute_surface_consistency(flame_params_offsets["offsets"])

        edge_loss = self._compute_edge_length_loss(offsets_verts)
        shape_reg, expr_reg, pose_reg = self._compute_flame_reg_losses(batch)

        loss_weights = self.get_current_lrs_n_lossweights()

        total_loss = loss_weights["w_norm"] * normal_loss + loss_weights["w_lap"] * lap_loss + \
                     loss_weights["w_silh"] * silh_loss + loss_weights["w_edge"] * edge_loss + \
                     loss_weights["w_lmk"] * lmk_loss + loss_weights["w_eye_closed"] * eye_closed_loss + \
                     loss_weights["w_shape_reg"] * shape_reg + loss_weights["w_expr_reg"] * expr_reg + \
                     loss_weights["w_pose_reg"] * pose_reg + loss_weights["w_surface_reg"] * surface_reg + \
                     loss_weights["w_semantic_ear"] * sem_ear + loss_weights["w_semantic_mouth"] * sem_mouth + \
                     loss_weights["w_semantic_hair"] * sem_hair + loss_weights["w_semantic_eye"] * sem_eye

        log_dict = {
            "lap_loss": lap_loss,
            "edge_loss": edge_loss,
            "lmk_loss": lmk_loss,
            "eye_closed_loss": eye_closed_loss,
            "norm_loss": normal_loss,
            "silh_loss": silh_loss,
            "shape_reg": shape_reg,
            "expr_reg": expr_reg,
            "pose_reg": pose_reg,
            "surface_reg": surface_reg,
            "iou": total_iou,
            "semantic_loss_ear": sem_ear,
            "semantic_iou_ear": iou_ear,
            "semantic_loss_eye": sem_eye,
            "semantic_iou_eye": iou_eye,
            "semantic_loss_mouth": sem_mouth,
            "semantic_iou_mouth": iou_mouth,
            "semantic_loss_hair": sem_hair,
            "semantic_iou_hair": iou_hair,
            "total_loss": total_loss,
        }

        for k, v in self._decays.items():
            decay = v.get(self.current_epoch)
            log_dict[f"decay_{k}"] = decay

        return total_loss, log_dict

    def _add_noise_to_flame_params(self, flame_params):
        """
        INPLACE noise adding to flame params 'rotation' 'neck' 'jaw' 'expr'
        :param flame_params: dict returned by self._create_flame_param_batch
        :return: dict with same structure as input but with parameter-specific noise added
        """
        noise_multiplier = dict(
            rotation=self.hparams["glob_rot_noise"] / 180 * np.pi,
            eyes=15.0 / 180 * np.pi,
            neck=0.15,
            jaw=0.2,
            expr=1.0,
        )

        for key in noise_multiplier:
            noise = torch.randn_like(flame_params[key]) * noise_multiplier[key] * self.hparams["flame_noise"]
            flame_params[key] = flame_params[key] + noise

        return flame_params

    def _optimize_texture(self, batch):

        # construct mesh
        flame_params_offsets = self._create_flame_param_batch(batch)
        offsets_verts, _ = self._forward_flame(flame_params_offsets)

        # produce noisy flame parameters for conditioning the texture (generalizes better)
        flame_params_offsets_detached = dict()
        for key, val in flame_params_offsets.items():
            flame_params_offsets_detached[key] = val.detach()
        flame_params_noise = self._add_noise_to_flame_params(flame_params_offsets_detached)
        _, _, mouth_conditioning = self._forward_flame(flame_params_noise, return_mouth_conditioning=True)
        expr = flame_params_noise["expr"]
        pose = torch.cat(
            (
                flame_params_noise["rotation"],
                flame_params_noise["neck"],
                flame_params_noise["jaw"],
                flame_params_noise["eyes"],
            ),
            dim=1,
        )

        phot_lossdict = self._compute_rgb_losses(
            batch,
            offsets_verts,
            expr=expr,
            pose=pose,
            mouth_conditioning=mouth_conditioning,
        )
        rgb_loss = phot_lossdict["rgb_loss"]
        perc_loss = phot_lossdict["perc_loss"]

        weights = self.get_current_lrs_n_lossweights()

        total_loss = weights["w_rgb"] * rgb_loss + weights["w_perc"] * perc_loss

        log_dict = {
            "rgb_loss": rgb_loss,
            "perc_loss": perc_loss,
            "total_loss": total_loss,
        }

        for k, v in self._decays.items():
            decay = v.get(self.current_epoch)
            log_dict[f"decay_{k}"] = decay

        return total_loss, log_dict

    def _optimize_jointly(self, batch):

        # construct mesh
        flame_params = self._create_flame_param_batch(batch, ignore_offsets=True)
        vertices, _ = self._forward_flame(flame_params)
        flame_params_offsets = self._create_flame_param_batch(batch)
        offsets_verts, pred_lmks = self._forward_flame(flame_params_offsets)
        raster_res = self._rasterize_flame(batch, offsets_verts)

        # produce noisy flame parameters for conditioning the texture (generalizes better)
        flame_params_offsets_detached = dict()
        for key, val in flame_params_offsets.items():
            flame_params_offsets_detached[key] = val.detach()
        flame_params_noise = self._add_noise_to_flame_params(flame_params_offsets_detached)
        _, _, mouth_conditioning = self._forward_flame(flame_params_noise, return_mouth_conditioning=True)

        expr = flame_params_noise["expr"]
        pose = torch.cat(
            (
                flame_params_noise["rotation"],
                flame_params_noise["neck"],
                flame_params_noise["jaw"],
                flame_params_noise["eyes"],
            ),
            dim=1,
        )

        # calculating losses
        sem_res = self._compute_semantic_loss(batch, offsets_verts, rasterized_results=raster_res,
                                              return_total_iou=True)
        semantic_loss, semantic_iou, semantic_pred, total_iou = sem_res
        sem_ear, sem_eye, sem_mouth, sem_hair = semantic_loss
        iou_ear, iou_eye, iou_mouth, iou_hair = semantic_iou

        normal_loss, rendered_normals = self._compute_normal_loss(
            batch,
            offsets_verts,
            return_normal_pred=True,
            rendered_semantics=semantic_pred,
            rasterized_results=raster_res,
        )

        phot_lossdict = self._compute_rgb_losses(
            batch,
            offsets_verts,
            expr=expr,
            pose=pose,
            mouth_conditioning=mouth_conditioning,
        )

        rgb_loss = phot_lossdict["rgb_loss"]
        perc_loss = phot_lossdict["perc_loss"]

        silh_loss = self._compute_silhouette_loss(batch, offsets_verts, rasterized_results=raster_res)

        eye_closed_loss = self._compute_eye_closed_loss(copy(flame_params), batch)

        edge_loss = self._compute_edge_length_loss(offsets_verts)
        lmk_loss = self._compute_lmk_loss(batch, pred_lmks)

        # regularization terms
        lap_loss = self._compute_laplacian_smoothing_loss(vertices, offsets_verts)
        surface_reg = self._compute_surface_consistency(flame_params_offsets["offsets"])
        shape_reg, expr_reg, pose_reg = self._compute_flame_reg_losses(batch)

        weights = self.get_current_lrs_n_lossweights()

        total_loss = weights["w_rgb"] * rgb_loss + weights["w_silh"] * silh_loss + \
                     weights["w_semantic_ear"] * sem_ear + weights["w_semantic_eye"] * sem_eye + \
                     weights["w_semantic_mouth"] * sem_mouth + weights["w_semantic_hair"] * sem_hair + \
                     weights["w_edge"] * edge_loss + weights["w_norm"] * normal_loss + weights["w_lap"] * lap_loss + \
                     weights["w_lmk"] * lmk_loss + weights["w_eye_closed"] * eye_closed_loss + \
                     weights["w_perc"] * perc_loss + weights["w_shape_reg"] * shape_reg + \
                     weights["w_surface_reg"] * surface_reg + weights["w_expr_reg"] * expr_reg + \
                     weights["w_pose_reg"] * pose_reg

        log_dict = {
            "rgb_loss": rgb_loss,
            "silh_loss": silh_loss,
            "norm_loss": normal_loss,
            "perc_loss": perc_loss,
            "lap_loss": lap_loss,
            "edge_loss": edge_loss,
            "lmk_loss": lmk_loss,
            "eye_closed_loss": eye_closed_loss,
            "shape_reg": shape_reg,
            "expr_reg": expr_reg,
            "surface_reg": surface_reg,
            "pose_reg": pose_reg,
            "iou": total_iou,
            "semantic_loss_ear": sem_ear,
            "semantic_iou_ear": iou_ear,
            "semantic_loss_eye": sem_eye,
            "semantic_iou_eye": iou_eye,
            "semantic_loss_mouth": sem_mouth,
            "semantic_iou_mouth": iou_mouth,
            "semantic_loss_hair": sem_hair,
            "semantic_iou_hair": iou_hair,
            "total_loss": total_loss,
        }

        for k, v in self._decays.items():
            decay = v.get(self.current_epoch)
            log_dict[f"decay_{k}"] = decay

        return total_loss, log_dict

    @torch.no_grad()
    def prepare_batch(self, batch):
        """
        prepares data obtained from dataloaders before they can be processed by NHAOptimizer during training. Not
        needed at inference time. Brings fg/bg segmentation map into right format, fills gt rgb image background with
        white, filters out noisy segmentations

        :param batch: data batch as provided from dataloader of RealDataModule.train_dataloader() in real.py
        :return: data batch dict
        """

        batch["seg"] = digitize_segmap(batch["seg"])
        batch["seg"] = batch["seg"].float()
        batch["rgb"] = fill_tensor_background(batch["rgb"], batch["seg"])

        if "lmk2d_iris" in batch:
            batch["lmk2d"] = torch.cat([batch["lmk2d"], batch["lmk2d_iris"]], dim=1)

        # decide if the segmentation maps can be trusted --> they are noisy for profile views
        # trust segmentation only up to 30 degrees away from front view
        gaze = torch.tensor([0, 0, -1.0], device=self.device)
        pose = batch["flame_pose"]
        global_rot = pose[:, :3]
        neck = self._flame._apply_rotation_limit(pose[:, 3:6], self._flame.neck_limits)
        B = len(neck)
        mats = batch_rodrigues(torch.cat([global_rot, neck], dim=0))
        gaze = mats[:B] @ mats[B:] @ gaze.view(3, 1)

        azi = torch.atan2(gaze[:, 2], gaze[:, 0]) / np.pi * 180
        azi -= 90  # zero centered
        azi_trust = torch.sigmoid(30 - torch.abs(azi))

        # elevation must be in range -np.pi/2 to np.pi/2
        ele = (torch.arccos(gaze[:, 1]) - np.pi / 2) / np.pi * 180
        ele_trust = torch.sigmoid(20 - torch.abs(ele))

        # trust = (azi > 60) & (azi < 120) & (ele > -20) & (ele < 20)
        trust = ele_trust * azi_trust
        batch["trust_semantics"] = trust.view(-1)
        return batch

    def toggle_optimizer(self, optimizers):
        """
        Makes sure only the gradients of the current optimizer's parameters are calculated
        in the training step to prevent dangling gradients in multiple-optimizer setup.

        .. note:: Only called when using multiple optimizers

        Override for your own behavior

        It works with ``untoggle_optimizer`` to make sure param_requires_grad_state is properly reset.

        Args:
            optimizer: Current optimizer used in training_loop
            optimizer_idx: Current optimizer idx in training_loop
        """

        # Iterate over all optimizer parameters to preserve their `requires_grad` information
        # in case these are pre-defined during `configure_optimizers`
        param_requires_grad_state = {}
        for opt in self.optimizers(use_pl_optimizer=False):
            for group in opt.param_groups:
                for param in group["params"]:
                    # If a param already appear in param_requires_grad_state, continue
                    if param in param_requires_grad_state:
                        continue
                    param_requires_grad_state[param] = param.requires_grad
                    param.requires_grad = False

        # Then iterate over the current optimizer's parameters and set its `requires_grad`
        # properties accordingly
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param.requires_grad = param_requires_grad_state[param]
        self._param_requires_grad_state = param_requires_grad_state

    def untoggle_optimizer(self):
        for param, requires_grad in self._param_requires_grad_state.items():
            param.requires_grad = requires_grad

        # save memory
        self._param_requires_grad_state = dict()

    def step(self, batch, batch_idx, stage="train"):
        if stage == "train" or self.fit_residuals:
            optim = self._get_current_optimizer()

            self.toggle_optimizer(optim)  # automatically toggle right requires_grads
            # step 1: standard optimization of flame model and offsets
            if self.current_epoch < self.hparams["epochs_offset"]:
                loss, log_dict = self._optimize_offsets(batch)

            # step 2: optimization texture
            elif self.current_epoch < self.hparams["epochs_offset"] + self.hparams["epochs_texture"]:
                loss, log_dict = self._optimize_texture(batch)

            # step 3: optimization texture and shape
            else:
                loss, log_dict = self._optimize_jointly(batch)

            # for opt in optim:
            #     opt.zero_grad()

            self.manual_backward(loss)

            for opt in optim:
                opt.step()

            for opt in optim:
                opt.zero_grad(set_to_none=True)  # saving gpu memory

            self.untoggle_optimizer()
        else:
            with torch.no_grad():
                if self.current_epoch < self.hparams["epochs_offset"]:
                    loss, log_dict = self._optimize_offsets(batch)
                else:
                    loss, log_dict = self._optimize_jointly(batch)

        # give all keys in logdict val_ prefix
        for key in list(log_dict.keys()):
            val = log_dict.pop(key)
            log_dict[f"{stage}_{key}"] = val

        # log scores
        log_dict["step"] = self.current_epoch
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # log images
        log_images = self.current_epoch % self.hparams["image_log_period"] == 0
        if batch_idx == 0 and log_images:
            try:
                interesting_frames = [1455, 536] if stage == "train" else [826, 1399]
                dataset = self.trainer.train_dataloader.dataset.datasets if stage == "train" else \
                    self.trainer.val_dataloaders[0].dataset
                interesting_samples = [dataset[dataset.frame_list.index(f)] for f in interesting_frames]
            except ValueError:
                interesting_samples = [dataset[i] for i in np.linspace(0, len(dataset), 4).astype(int)[1:3]]
            vis_batch = dict_2_device(stack_dicts(*interesting_samples), self.device)
            vis_batch = self.prepare_batch(vis_batch)
            self._visualize_head(vis_batch, max_samples=2, title=stage)

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        self.is_train = True
        self.fit_residuals = False
        self.prepare_batch(batch)
        ret = self.step(batch, batch_idx, stage="train")
        self.is_train = False
        return ret

    def validation_step(self, batch, batch_idx, **kwargs):
        self.fit_residuals = self.current_epoch < self.hparams["epochs_offset"] or self.current_epoch >= self.hparams[
            "epochs_texture"] + self.hparams["epochs_offset"]
        batch = self.prepare_batch(batch)

        with torch.set_grad_enabled(self.fit_residuals):
            self.step(batch, batch_idx, stage="val")
        self.fit_residuals = False

    @torch.no_grad()
    def _visualize_head(self, batch, max_samples=5, title=f"flame_fit"):
        rgba_pred = self.forward(batch, symmetric_rgb_range=False)
        shaded_pred = self.predict_shaded_mesh(batch)

        N = min(max_samples, batch["rgb"].shape[0])

        rgb_gt = batch["rgb"]
        rgb_gt = rgb_gt[:N] * 0.5 + 0.5
        rgb_pred = rgba_pred[:N, :3]
        shaded_pred = shaded_pred[:N, :3]

        images = torch.cat([rgb_gt, rgb_pred, shaded_pred], dim=0)
        log_img = torchvision.utils.make_grid(images, nrow=N)
        self.logger.experiment.add_image(title + "prediction", log_img, self.current_epoch)

    def forward(self, batch, ignore_expr=False, ignore_pose=False, center_prediction=False, symmetric_rgb_range=True):
        """
        returns rgba tensor of shape N x 3+1 x H x W with rgb values ranging from -1 ... 1 and alpha value
        ranging from 0 to +1
        :param batch:
        :param symmetric_rgb_range: if True rgb values range from -1 ... 1 else 0 ... 1
        :return:
        """
        K = batch["cam_intrinsic"]
        RT = batch["cam_extrinsic"]
        H, W = batch["rgb"].shape[-2:]

        flame_params_offsets = self._create_flame_param_batch(batch, ignore_expr=ignore_expr, ignore_pose=ignore_pose)
        offsets_verts, _, mouth_conditioning = self._forward_flame(flame_params_offsets, return_mouth_conditioning=True)

        # rgba prediction
        expr = flame_params_offsets["expr"]
        pose = torch.cat((flame_params_offsets["rotation"], flame_params_offsets["neck"], flame_params_offsets["jaw"],
                          flame_params_offsets["eyes"]), dim=1)

        rgba_pred = self._render_rgba(offsets_verts, K, RT, H, W, expr=expr, pose=pose, mouth_cond=mouth_conditioning,
                                      center_prediction=center_prediction)

        if not symmetric_rgb_range:
            rgba_pred[:, :3] = torch.clip(rgba_pred[:, :3] * 0.5 + 0.5, min=0.0, max=1.0)

        return rgba_pred

    @torch.no_grad()
    def predict_reenaction(self, batch, driving_model, base_target_params, base_driving_params, return_alpha=False):

        K = batch["cam_intrinsic"]
        RT = batch["cam_extrinsic"]
        N, C, H, W = batch["rgb"].shape

        # OBTAINING FLAME PARAMS
        flame_params_offsets = self._create_flame_param_batch(batch)

        # insert correct shape parameters
        flame_params_offsets["shape"] = base_target_params["shape"].expand(N, -1)

        # adopt driving frame parameters
        flame_params_driving = driving_model._create_flame_param_batch(batch)
        for key in ["expr", "translation", "rotation", "neck", "jaw", "eyes"]:
            residual_param = flame_params_driving[key] - base_driving_params[key].expand(N, -1)
            flame_params_offsets[key] = base_target_params[key].expand(N, -1) + residual_param

        offsets_verts, _, mouth_conditioning = self._forward_flame(flame_params_offsets, return_mouth_conditioning=True)

        # rgba prediction
        expr = flame_params_offsets["expr"]
        pose = torch.cat((flame_params_offsets["rotation"], flame_params_offsets["neck"], flame_params_offsets["jaw"],
                          flame_params_offsets["eyes"]), dim=1)

        rgba_pred = self._render_rgba(offsets_verts, K, RT, H, W, expr=expr, pose=pose, mouth_cond=mouth_conditioning)

        rgb_pred, seg_pred = rgba_pred[:, :3], rgba_pred[:, 3:]

        # rgb pred logging
        # rgb_pred.permute(0, 2, 3, 1)[seg_pred[:, 0] < .5] = 1  # change background to white  NOT NECESSARY
        rgb_pred = torch.clip(rgb_pred * 0.5 + 0.5, min=0, max=1)
        if return_alpha:
            return rgb_pred, seg_pred
        else:
            return rgb_pred

    @torch.no_grad()
    def predict_shaded_mesh(self, batch, tex_color=np.array((188, 204, 245)) / 255, light_colors=(0.4, 0.6, 0.3)):
        K = batch["cam_intrinsic"]
        RT = batch["cam_extrinsic"]
        H, W = batch["rgb"].shape[-2:]

        flame_params_offsets = self._create_flame_param_batch(batch)
        offsets_verts, _ = self._forward_flame(flame_params_offsets)

        vertex_colors = torch.ones_like(offsets_verts) * torch.tensor(tex_color, device=self.device).float().view(1, 3)
        # define meshes and textures
        tex = TexturesVertex(vertex_colors)
        mesh = Meshes(
            verts=offsets_verts,
            faces=self._flame.faces[None].expand(len(offsets_verts), -1, -1),
            textures=tex,
        )

        return render_shaded_mesh(mesh, K, RT, (H, W), self.device, light_colors)

    def get_current_lrs_n_lossweights(self):
        epoch = self.current_epoch

        if epoch < self.hparams["epochs_offset"]:
            i = 0

        elif epoch < self.hparams["epochs_offset"] + self.hparams["epochs_texture"]:
            i = 1

        else:
            i = 2

        return dict(
            w_rgb=self.hparams["w_rgb"][i],
            w_perc=self.hparams["w_perc"][i],
            w_norm=self.hparams["w_norm"][i],
            w_edge=self.hparams["w_edge"][i],
            w_eye_closed=self.hparams["w_eye_closed"][i],
            w_semantic_ear=self.hparams["w_semantic_ear"][i],
            w_semantic_eye=self.hparams["w_semantic_eye"][i],
            w_semantic_mouth=self.hparams["w_semantic_mouth"][i],
            w_semantic_hair=self._decays["semantic"].get(epoch),
            w_silh=self._decays["silh"].get(epoch),
            w_lap=self._decays["lap"].get(epoch),
            w_surface_reg=self.hparams["w_surface_reg"][i],
            w_lmk=self.hparams["w_lmk"][i],
            w_shape_reg=self.hparams["w_shape_reg"][i],
            w_expr_reg=self.hparams["w_expr_reg"][i],
            w_pose_reg=self.hparams["w_pose_reg"][i],
            texture_weight_decay=self.hparams["texture_weight_decay"][i],
            flame_lr=self.hparams["flame_lr"][i],
            offset_lr=self.hparams["offset_lr"][i],
            tex_lr=self.hparams["tex_lr"][i],
            trans_lr=self._trans_lr[i],
        )

    def configure_optimizers(self):

        # FLAME
        flame_params = [self._shape, self._expr, self._rotation, self._jaw_pose, self._neck_pose]

        lrs = self.get_current_lrs_n_lossweights()

        # translation gets smaller learning rate
        params = [
            {"params": flame_params},
            {"params": [self._translation], "lr": lrs["trans_lr"]},
        ]
        flame_optim = torch.optim.SGD(params, lr=lrs["flame_lr"])

        # OFFSETS
        params = [{"params": self._vert_feats}]
        params.append({"params": self._offset_mlp.parameters()})
        offset_optim = torch.optim.Adam(params, lr=lrs["offset_lr"])

        # TEXTURE optimizer
        params = [
            {"params": list(self._texture.parameters()) + list(self._normal_encoder.parameters())},
            {"params": self._explFeatures.parameters()},
        ]
        tex_optim = torch.optim.Adam(params, lr=lrs["tex_lr"], weight_decay=lrs["texture_weight_decay"])

        # JOINT FLAME
        joint_flame_params = flame_params + [self._eyes_pose]
        # translation gets smaller lr
        params = [
            {"params": joint_flame_params},
            {"params": [self._translation], "lr": lrs["trans_lr"]},
        ]
        joint_flame_optim = torch.optim.SGD(params, lr=lrs["flame_lr"])

        # RESIDUALS optimizer
        resid_params = [self._expr, self._rotation, self._jaw_pose, self._neck_pose]
        params = [
            {"params": resid_params},
            {"params": [self._translation], "lr": lrs["trans_lr"]},
        ]
        offset_resid_optim = torch.optim.SGD(params, lr=lrs["flame_lr"])

        params = [
            {"params": resid_params + [self._eyes_pose]},
            {"params": [self._translation], "lr": lrs["trans_lr"]},
        ]
        all_resid_optim = torch.optim.SGD(params, lr=lrs["flame_lr"])  # adds eye rotations

        return [
            {"optimizer": flame_optim},
            {"optimizer": offset_optim},
            {"optimizer": tex_optim},
            {"optimizer": joint_flame_optim},
            {"optimizer": offset_resid_optim},
            {"optimizer": all_resid_optim},
        ]
