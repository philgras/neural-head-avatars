"""
Code heavily inspired by https://github.com/HavenFeng/photometric_optimization/blob/master/models/FLAME.py. 
Please consider citing their work if you find this code useful. The code is subject to the license available via
https://github.com/vchoutas/smplx/edit/master/LICENSE

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

"""
import sys

sys.path.append('./deps')

from nha.util.lbs import lbs, batch_rodrigues, vertices2landmarks
from nha.util.meshes import face_vertices, vertex_normals
from nha.util.log import get_logger
from nha.util.meshes import edge_subdivide
from pytorch3d.io import load_obj, save_obj
from collections import OrderedDict
from scipy.spatial.transform import Rotation
from pytorch3d.structures import Meshes

import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F

logger = get_logger(__name__)

FLAME_MODEL_PATH = 'assets/flame/generic_model.pkl'
FLAME_MESH_MOUTH_PATH = 'assets/flame/head_template_mesh_mouth.obj'
FLAME_PARTS_MOUTH_PATH = 'assets/flame/FLAME_masks_mouth.pkl'
FLAME_LMK_PATH = 'assets/flame/landmark_embedding_with_eyes.npy'
FLAME_LOWER_NECK_FACES_PATH = 'assets/flame/lower_neck_face_idcs.npy'
FLAME_N_SHAPE = 300
FLAME_N_EXPR = 100


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


class FlameHead(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self,
                 shape_params,
                 expr_params,
                 flame_model_path=FLAME_MODEL_PATH,
                 flame_lmk_embedding_path=FLAME_LMK_PATH,
                 flame_template_mesh_path=FLAME_MESH_MOUTH_PATH,
                 flame_parts_path=FLAME_PARTS_MOUTH_PATH,
                 eye_limits=((-50, 50), (-50, 50), (-0.1, 0.1)),
                 neck_limits=((-90, 90), (-60, 60), (-80, 80)),
                 jaw_limits=((-5, 60), (-0.1, 0.1), (-0.1, 0.1)),
                 global_limits=((-20, 20), (-90, 90), (-20, 20)),
                 ignore_faces=[],
                 upsample_regions=OrderedDict(),
                 spatial_blur_sigma=0.01
                 ):
        super().__init__()

        global_limits = torch.tensor(global_limits).float() / 180 * np.pi
        self.register_buffer('global_limits', global_limits)
        neck_limits = torch.tensor(neck_limits).float() / 180 * np.pi
        self.register_buffer('neck_limits', neck_limits)
        jaw_limits = torch.tensor(jaw_limits).float() / 180 * np.pi
        self.register_buffer('jaw_limits', jaw_limits)
        eye_limits = torch.tensor(eye_limits).float() / 180 * np.pi
        self.register_buffer('eye_limits', eye_limits)

        self._ignore_faces = ignore_faces
        self.n_shape_params = shape_params
        self.n_expr_params = expr_params

        with open(flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        # The vertices of the template model
        self.register_buffer('_v_template',
                             to_tensor(to_np(flame_model.v_template), dtype=self.dtype))

        # add faces and uvs
        _, faces, aux = load_obj(flame_template_mesh_path, load_textures=False)
        self.register_buffer('_faces', faces.verts_idx, persistent=False)
        self.register_buffer('_faces_uvs', faces.textures_idx, persistent=False)
        self.register_buffer('_vertex_uvs', aux.verts_uvs, persistent=False)

        # load vertex lists corresponding to parts of the face
        with open(flame_parts_path, 'rb') as f:
            parts = pickle.load(f, encoding='latin1')
            self._parts = OrderedDict()
            for key in sorted(parts.keys()):
                self._parts[key] = torch.tensor(parts[key])

        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, :shape_params],
                               shapedirs[:, :, 300:300 + expr_params]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer('J_regressor',
                             to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(flame_lmk_embedding_path, allow_pickle=True,
                                 encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_embeddings['static_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_embeddings['static_lmk_bary_coords'],
                                          dtype=self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx',
                             lmk_embeddings['dynamic_lmk_faces_idx'].long())
        self.register_buffer('dynamic_lmk_bary_coords',
                             lmk_embeddings['dynamic_lmk_bary_coords'].float())
        self.register_buffer('full_lmk_faces_idx',
                             torch.tensor(lmk_embeddings['full_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('full_lmk_bary_coords',
                             torch.tensor(lmk_embeddings['full_lmk_bary_coords'], dtype=self.dtype))

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

        self.register_buffer("spatially_blurred_vert_labels",
                             self._get_spatially_blurred_vert_labels(spatial_blur_sigma),
                             persistent=False)

        # performing upsampling IMPORTANT TO DO IT HERE WHERE ALL ASSETS ARE LOADED ALREADY
        # BUT NO DERIVED QUANTITIES ARE CALCULATED YET
        self._upsample_regions(upsample_regions)
        # self._upsample_faces(list(range(len(self._faces))))

        ###
        # calculating derived attributes
        ###
        v_temp = self._v_template
        v_min = v_temp.min(dim=0, keepdim=True).values
        v_max = v_temp.max(dim=0, keepdim=True).values
        v_temp = (v_temp - v_min) / (v_max - v_min)
        v_temp = 2 * (v_temp - 0.5)
        self.register_buffer('_v_template_normed', v_temp, persistent=False)

        self.register_buffer('_face_coords', self._v_template[self._faces], persistent=False)

        faces_coords_normed = self._face_coords - self._face_coords.flatten(end_dim=1).min(
            dim=0).values.view(1, 1, 3)
        faces_coords_normed /= faces_coords_normed.flatten(end_dim=1).max(dim=0).values.view(1, 1,
                                                                                             3)
        faces_coords_normed = faces_coords_normed * 2 - 1
        self.register_buffer('_face_coords_normed', faces_coords_normed, persistent=False)

        # create uvcoords per face --> this is what you can use for uv map rendering
        # range from -1 to 1 (-1, -1) = left top; (+1, +1) = right bottom
        # pad 1 to the end
        uvcoords = torch.cat([self._vertex_uvs, torch.ones(self._vertex_uvs.shape[0], 1)], -1)
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_verts = face_vertices(uvcoords[None, ...], self._faces_uvs[None, ...])
        self.register_buffer('_face_uvcoords', face_verts[0], persistent=False)

        # create a uvmap index for every vertex (0 index is flame head uv, 1 index is teeth uv)
        vertex_uvmap = torch.zeros(len(self._v_template)).long()
        zero_index = torch.tensor([]).long()
        vertex_uvmap[torch.cat([self._parts.get('mouth', zero_index),
                                self._parts.get('teeth', zero_index)])] = 1
        face_uvmap = vertex_uvmap[self._faces.view(-1)].view(self._faces.shape)
        face_uvmap = face_uvmap.min(dim=1)[0]
        self.register_buffer('_face_uvmap', face_uvmap, persistent=False)

        # storing normals
        template_normals = vertex_normals(self._v_template[None], self._faces[None])[0]
        self.register_buffer("_vertex_normals", template_normals, persistent=False)
        self.register_buffer("_face_normals", self._vertex_normals[self._faces], persistent=False)

        # DROPPING FACES IF REQUESTED (also drops unused vertices)
        if len(self._ignore_faces) != 0:
            filter_dict = self._get_vertNface_filters(self._ignore_faces)
            vert_reindexer = filter_dict["vert_reindexer"]
            uv_reindexer = filter_dict["uv_reindexer"]
            self.register_buffer("_vert_filter", filter_dict["vert_filter"], persistent=False)
            self.register_buffer("_face_filter", filter_dict["face_filter"], persistent=False)
            self.register_buffer("_uv_filter", filter_dict["uv_filter"], persistent=False)
            faces_filtered = vert_reindexer[self._faces[self._face_filter]]
            faces_uvs_filtered = uv_reindexer[self._faces_uvs[self._face_filter]]
            verts_filtered = self._v_template[self._vert_filter]
            uvs_filtered = self._vertex_uvs[self._uv_filter]
            self.register_buffer("_v_template_filtered", verts_filtered, persistent=False)
            self.register_buffer("_v_template_normed_filtered",
                                 self._v_template_normed[self._vert_filter], persistent=False)
            self.spatially_blurred_vert_labels = self.spatially_blurred_vert_labels[filter_dict["vert_filter"]]
            self.register_buffer("_faces_filtered", faces_filtered, persistent=False)
            self.register_buffer("_faces_uvs_filtered", faces_uvs_filtered, persistent=False)
            self.register_buffer("_uvs_filtered", uvs_filtered, persistent=False)
            self._parts_filtered = OrderedDict()
            for key, verts in self._parts.items():
                verts_filtered = torch.tensor(list(set(verts) - filter_dict["ignored_verts"]))
                self._parts_filtered[key] = vert_reindexer[verts_filtered]

        # semantic vert_labels
        self.register_buffer("vert_labels", torch.zeros(len(self.v_template),
                                                        len(self.get_body_parts()),
                                                        dtype=torch.float), persistent=False)
        for i, (key, verts) in enumerate(self.get_body_parts().items()):
            self.vert_labels[verts, i] = 1.

    @property
    def v_template(self):
        return self._v_template_filtered if len(self._ignore_faces) != 0 else self._v_template

    @property
    def v_template_normed(self):
        return self._v_template_normed_filtered if len(self._ignore_faces) != 0 \
            else self._v_template_normed

    @property
    def faces(self):
        return self._faces_filtered if len(self._ignore_faces) != 0 else self._faces

    @property
    def face_coords(self):
        return self._face_coords[self._face_filter] if len(self._ignore_faces) != 0 \
            else self._face_coords

    @property
    def face_coords_normed(self):
        return self._face_coords_normed[self._face_filter] if len(
            self._ignore_faces) != 0 else self._face_coords_normed

    @property
    def faces_uvs(self):
        return self._faces_uvs_filtered if len(self._ignore_faces) != 0 else self._faces_uvs

    @property
    def face_uvcoords(self):
        return self._face_uvcoords[self._face_filter] if len(
            self._ignore_faces) != 0 else self._face_uvcoords

    @property
    def face_uvmap(self):
        return self._face_uvmap[self._face_filter] if len(
            self._ignore_faces) != 0 else self._face_uvmap

    @property
    def face_normals(self):
        return self._face_normals[self._face_filter] if len(
            self._ignore_faces) != 0 else self._face_normals

    @property
    def vertex_uvs(self):
        return self._vertex_uvs[self._uv_filter] if len(
            self._ignore_faces) != 0 else self._vertex_uvs

    @property
    def vertex_normals(self):
        return self._vertex_normals[self._vert_filter] if len(
            self._ignore_faces) != 0 else self._vertex_normals

    def _get_spatially_blurred_vert_labels(self, sigma=0.01):
        """
        calculates spatially blurred semantic vertex labels of shape V x C

        uses unmodified template mesh (due to memory constraints) so make sure to upsample result when upsampling mesh

        :return: V x C with scores between 0 and 1.
        """
        UPPER_PLACEHOLDER = 1

        meshes = Meshes(verts=[self._v_template], faces=[self._faces])
        verts = meshes.verts_packed()
        edges = meshes.edges_packed()
        V = verts.shape[0]

        # initializing distance table
        distances = torch.zeros(V, V, dtype=verts.dtype, device=verts.device) + UPPER_PLACEHOLDER
        distances[torch.arange(V), torch.arange(V)] = 0
        e1, e2 = edges.unbind(1)
        distances[e1, e2] = distances[e2, e1] = torch.linalg.norm(verts[e1] - verts[e2], dim=-1, ord=2)
        max_neighbours = (torch.mean(torch.sum(distances < UPPER_PLACEHOLDER, dim=-1).float() - 1) * 2).int()
        assert torch.max(distances) == UPPER_PLACEHOLDER  # check if upper placeholder was set high enough

        # iteratively propagate distances to linked vertices
        v_helper = torch.arange(V).view(V, 1).expand(V, max_neighbours)
        for i in range(5):
            neighbours = torch.argsort(distances, dim=-1)[:, :max_neighbours]  # V x neighb
            distances = torch.min(distances[neighbours] + distances[v_helper, neighbours].unsqueeze(-1), dim=1).values

        weights = torch.softmax((-(distances ** 2) / (2 * sigma ** 2)), dim=-1)

        vert_labels = torch.zeros(V, len(self._parts), dtype=torch.float, device=self._v_template.device)
        for i, (key, verts) in enumerate(self._parts.items()):
            vert_labels[verts, i] = 1.

        blurred_semantics = torch.matmul(weights, vert_labels)  # (V, C)

        return blurred_semantics

    def _get_vertNface_filters(self, ignored_faces):
        device = self._faces.device

        # filtered faces
        face_filter = torch.ones(len(self._faces)).bool()
        face_filter[ignored_faces] = False
        new_faces = self._faces[face_filter]
        new_face_uvs = self._faces_uvs[face_filter]

        # vertex filter
        vertex_filter = torch.zeros(len(self._v_template)).bool()
        vertex_filter[new_faces] = True
        ignored_verts = set(range(len(self._v_template))) - set(
            torch.unique(new_faces).numpy().astype(int))

        # uv filter
        uv_filter = torch.zeros(len(self._vertex_uvs)).bool()
        uv_filter[new_face_uvs] = True
        ignored_uvs = set(range(len(self._vertex_uvs))) - set(
            torch.unique(new_face_uvs).numpy().astype(int))

        oldvertidx2newvertidx = torch.tensor(
            [vertex_filter[:i + 1].long().sum() - 1 for i in range(len(vertex_filter))],
            device=device)
        olduvidx2newuvidx = torch.tensor(
            [uv_filter[:i + 1].long().sum() - 1 for i in range(len(uv_filter))],
            device=device)
        oldfaceidx2newfaceidx = torch.tensor(
            [face_filter[:i + 1].sum() - 1 for i in range(len(face_filter))],
            device=device)

        return dict(vert_filter=vertex_filter, face_filter=face_filter, uv_filter=uv_filter,
                    vert_reindexer=oldvertidx2newvertidx,
                    face_reindexer=oldfaceidx2newfaceidx,
                    uv_reindexer=olduvidx2newuvidx,
                    ignored_verts=ignored_verts,
                    ignored_uvs=ignored_uvs)

    def _upsample_regions(self, upsample_regions: OrderedDict):
        """
        :param upsample_regions: dict
                                "face_region": upsample_factor (int)
        :return:
        """

        if len(upsample_regions) == 0:
            return

        for key, val in upsample_regions.items():
            if key != "all" and val > 0:
                raise NotImplementedError("Seperated upsampling of different regions not implemented.")

        if "all" in upsample_regions and upsample_regions["all"] > 0:
            for i in range(upsample_regions["all"]):
                eye_vert_idcs = self.get_body_part_vert_idcs(*[part for part in self._parts if "eyeball" in part],
                                                             consider_full_flame_model=True, )
                eye_face_idcs = self.faces_of_verts(eye_vert_idcs,
                                                    consider_full_flame_model=True,
                                                    return_face_idcs=True)[1]
                face_idcs = list(set(range(len(self._faces))) - set(self._ignore_faces) - set(eye_face_idcs))
                self._upsample_faces(face_idcs)

        # max_count = max([v for v in upsample_regions.values()])
        # for i in range(max_count):
        #     face_idcs = []
        #     for part, upsample_count in upsample_regions.items():
        #         if i < upsample_count:
        #             _, faces = self.faces_of_verts(self.get_body_part_vert_idcs(part, consider_full_flame_model=True),
        #                                            consider_full_flame_model=True, return_face_idcs=True)
        #             face_idcs += faces
        #     self._upsample_faces(face_idcs)

    def _upsample_faces(self, face_idcs):
        """
        splits every given face into 3 subfaces and performs necessary adjustments to flame model
        :param face_idcs: index list of length F'
        :return: 
        """
        face_idcs = list(np.unique(face_idcs))
        n_v = len(self._v_template)
        n_t = len(self._vertex_uvs)
        n_f = len(self._faces)

        verts, uvs, faces, uv_faces, edges, uv_edges \
            = edge_subdivide(vertices=self._v_template.cpu().numpy(),
                             uvs=self._vertex_uvs.cpu().numpy(),
                             faces=self._faces[face_idcs].cpu().numpy(),
                             uvfaces=self._faces_uvs[face_idcs].cpu().numpy())
        faces = torch.cat((self._faces, torch.tensor(faces[len(face_idcs):],
                                                     dtype=self._faces.dtype,
                                                     device=self._faces.device)), dim=0)
        uv_faces = torch.cat((self._faces_uvs, torch.tensor(uv_faces[len(face_idcs):],
                                                            dtype=self._faces_uvs.dtype,
                                                            device=self._faces_uvs.device)), dim=0)
        n_edges = len(edges)

        self.register_buffer("_v_template", torch.tensor(verts, dtype=self._v_template.dtype),
                             persistent=False)
        self.register_buffer("_vertex_uvs", torch.tensor(uvs, dtype=self._vertex_uvs.dtype),
                             persistent=False)
        self.register_buffer("_faces", faces, persistent=False)
        self.register_buffer("_faces_uvs", uv_faces, persistent=False)

        # calculate new blendshapes
        new_shapedirs = self.shapedirs[edges]  # n_edges x 2 x 3 x 400
        new_shapedirs = new_shapedirs.mean(dim=1)  # n_edges x 3 x 400
        new_posedirs = self.posedirs.permute(1, 0).view(n_v, 3, 36)  # V x 3 x 36
        new_posedirs = new_posedirs[edges]  # n_edges x 2 x 3 x 36
        new_posedirs = new_posedirs.mean(dim=1)  # n_edges x 3 x 36
        new_posedirs = new_posedirs.view(n_edges * 3, 36).permute(1, 0)  # 36 x n_edges * 3
        self.register_buffer("shapedirs", torch.cat((self.shapedirs, new_shapedirs), dim=0),
                             persistent=False)
        self.register_buffer("posedirs", torch.cat((self.posedirs, new_posedirs), dim=1),
                             persistent=False)

        # calculate new lbs components
        new_J_regressor = torch.zeros(5, n_edges).to(self.J_regressor.dtype).to(self.J_regressor.device)
        new_lbs_weights = self.lbs_weights[edges]  # n_edges x 2 x 5
        new_lbs_weights = new_lbs_weights.mean(dim=1)  # n_edges x 5
        self.register_buffer("J_regressor", torch.cat((self.J_regressor, new_J_regressor), dim=1),
                             persistent=False)
        self.register_buffer("lbs_weights", torch.cat((self.lbs_weights, new_lbs_weights), dim=0),
                             persistent=False)

        # update body parts
        for part, idcs in self._parts.items():
            new_vert_idcs = []
            for i in range(len(edges)):
                if edges[i, 0] in idcs and edges[i, 1] in idcs:
                    new_vert_idcs.append(i + n_v)
            if new_vert_idcs:
                new_vert_idcs = torch.tensor(new_vert_idcs, dtype=idcs.dtype, device=idcs.device)
                self._parts[part] = torch.cat((idcs, new_vert_idcs), dim=0)

        # ignore new faces if parent face was to be ignored
        ignored_upsampled_faces = list(set.intersection(set(self._ignore_faces), set(face_idcs)))
        if ignored_upsampled_faces:
            # each parent face produces 4 child faces that are stacked on top of face stack. child faces always stick
            # together (see edge_subdivide()). So idcs of child face of parent face with index i are given by:
            # c0 = n_faces + i * 4 + 0, c1 = n_faces + i*4 + 1, ...
            ignored_upsampled_faces_idcs = np.array([face_idcs.index(f) for f in ignored_upsampled_faces])
            ignored_upsampled_faces_idcs = n_f + ignored_upsampled_faces_idcs * 4
            new_ignored_faces = np.concatenate((ignored_upsampled_faces_idcs,
                                                ignored_upsampled_faces_idcs + 1,
                                                ignored_upsampled_faces_idcs + 2,
                                                ignored_upsampled_faces_idcs + 3), axis=0)
            new_ignored_faces = list(new_ignored_faces)
        else:
            new_ignored_faces = []

        # ignore old faces
        self._ignore_faces = list(set.union(set(self._ignore_faces), set(face_idcs), set(new_ignored_faces)))

        # calculate blurred vert labels
        new_spatially_blurred_vert_labels = self.spatially_blurred_vert_labels[edges].mean(dim=1)
        self.spatially_blurred_vert_labels = torch.cat((self.spatially_blurred_vert_labels,
                                                        new_spatially_blurred_vert_labels), dim=0)

    def save_2_obj(self, path,
                   shape=None,
                   expr=None,
                   rotation=None,
                   neck=None,
                   jaw=None,
                   eyes=None,
                   use_rotation_limits=True):
        if shape is None:
            shape = torch.zeros(1, 300, device=self._v_template.device, dtype=self._v_template.dtype)
        if expr is None:
            expr = torch.zeros(1, 100, device=self._v_template.device, dtype=self._v_template.dtype)
        if rotation is None:
            rotation = torch.zeros(1, 3, device=self._v_template.device, dtype=self._v_template.dtype)
        if neck is None:
            neck = torch.zeros(1, 3, device=self._v_template.device, dtype=self._v_template.dtype)
        if jaw is None:
            jaw = torch.zeros(1, 3, device=self._v_template.device, dtype=self._v_template.dtype)
        if eyes is None:
            eyes = torch.zeros(1, 6, device=self._v_template.device, dtype=self._v_template.dtype)

        from nha.util.general import write_obj
        verts = self.forward(shape.view(1, 300), expr.view(1, 100), rotation.view(1, 3), neck.view(1, 3),
                             jaw.view(1, 3), eyes.view(1, 6), return_landmarks=None,
                             use_rotation_limits=use_rotation_limits)
        write_obj(path, verts[0], self.faces, self.faces_uvs, self.vertex_uvs)

    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def select_3d68(self, vertices):
        landmarks3d = vertices2landmarks(vertices, self._faces,
                                         self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
                                         self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        return landmarks3d

    def get_body_parts(self, consider_full_flame_model=False):
        """
        returns body part vertex index list dictionary
        - keys: body part names
        - values: list of vertex idcs

        If consider_full_flame_model is True: returns vertex idcs of unfiltered mesh, otherwise applies filters as
        specified via self._ignore_faces to vertices and faces and returns body parts dict with corrected vertex idcs

        :param consider_full_flame_model:
        :return:
        """
        return self._parts_filtered if len(
            self._ignore_faces) != 0 and not consider_full_flame_model else self._parts

    def get_body_part_vert_idcs(self, *parts, consider_full_flame_model=False):
        """
        returns vertex indices that are part of list of specified body parts (strings). If '-' is added at beginning of
        body part name (e.g. '-face'), excludes body part.

        ATTENTION: part list is not commutative if there are overlaps between the
        body parts and at least one body part has prefix '-'

        If 'consider_full_flame_model' is True: returns vertex idcs of unfiltered mesh, otherwise applies filters as
        specified via self._ignore_faces to vertices and faces and returns body parts dict with corrected vertex idcs

        :param parts:
        :return: torch tensor with vertex idcs
        """
        part_dict = self.get_body_parts(consider_full_flame_model=consider_full_flame_model)
        ret = set()
        for p in parts:
            if p[0] == '-':
                ret = ret - set(part_dict[p[1:]].tolist())
            else:
                ret = set.union(ret, part_dict[p].tolist())
        ret = torch.tensor(sorted(ret), device=self._v_template.device)
        # sorting is most likely not necessary but done here to ensure deterministic order of idcs
        return ret

    def find_body_part_expr_params(self, *parts):
        """
        returns the array of idcs of expression parameters that affect body parts most.
        Sorted in descending order (first idx corresponds to expr. param with biggest impact)
        :param parts: same convention as in get_body_part_vert_idcs()
        :return: torch tensor of len 100 with argsorted expression param idcs
        """

        vert_idcs = self.get_body_part_vert_idcs(*parts, consider_full_flame_model=True)
        deviations = self.shapedirs[vert_idcs].norm(p=2, dim=1).mean(dim=0)
        return torch.argsort(deviations[self.n_shape_params:], descending=True)

    def faces_of_verts(self, vert_idcs, consider_full_flame_model=False, return_face_idcs=False):
        """
        calculates face tensor of shape F x 3 with face spanned by vertices in flame mesh
        all vertices of the faces returned by this function contain only vertices from vert_idcs
        :param vert_idcs:
        :param consider_full_flame_model:
        :return_face_idcs: if True, also returns list of relevant face idcs
        :return:
        """
        all_faces = self._faces if consider_full_flame_model else self._faces_filtered  # F x 3
        vert_idcs = vert_idcs.to(all_faces.device)
        vert_faces = []
        face_idcs = []
        for i, f in enumerate(all_faces):
            keep_face = True
            for idx in f:
                if not idx in vert_idcs:
                    keep_face = False
            if keep_face:
                vert_faces.append(f)
                face_idcs.append(i)
        vert_faces = torch.stack(vert_faces)

        if return_face_idcs:
            return vert_faces, face_idcs

        return vert_faces

    def _apply_rotation_limit(self, rotation, limit):
        r_min, r_max = limit[:, 0].view(1, 3), limit[:, 1].view(1, 3)
        diff = r_max - r_min
        return r_min + (torch.tanh(rotation) + 1) / 2 * diff

    def apply_rotation_limits(self, neck=None, jaw=None):
        """
        method to call for applying rotation limits. Don't use _apply_rotation_limit() in other methods as this
        might cause some bugs if we change which poses are affected by rotation limits. For this reason, in this method,
        all affected poses are limited within one function so that if we add more restricted poses, they can just be
        updated here
        :param neck:
        :param jaw:
        :return:
        """
        neck = self._apply_rotation_limit(neck, self.neck_limits) if neck is not None else None
        jaw = self._apply_rotation_limit(jaw, self.jaw_limits) if jaw is not None else None

        ret = [i for i in [neck, jaw] if i is not None]

        return ret[0] if len(ret) == 1 else ret

    def _revert_rotation_limit(self, rotation, limit):
        """
        inverse function of _apply_rotation_limit()
        from rotation angle vector (rodriguez) -> scalars from -inf ... inf
        :param rotation: tensor of shape N x 3
        :param limit: tensor of shape 3 x 2 (min, max)
        :return:
        """
        r_min, r_max = limit[:, 0].view(1, 3), limit[:, 1].view(1, 3)
        diff = r_max - r_min
        rotation = rotation.clone()
        for i in range(3):
            rotation[:, i] = torch.clip(rotation[:, i],
                                        min=r_min[0, i] + diff[0, i] * .01,
                                        max=r_max[0, i] - diff[0, i] * .01)
        return torch.atanh((rotation - r_min) / diff * 2 - 1)

    def revert_rotation_limits(self, neck, jaw):
        """
        inverse function of apply_rotation_limits()
        from rotation angle vector (rodriguez) -> scalars from -inf ... inf
        :param rotation:
        :param limit:
        :return:
        """
        neck = self._revert_rotation_limit(neck, self.neck_limits)
        jaw = self._revert_rotation_limit(jaw, self.jaw_limits)
        return neck, jaw

    def get_neutral_joint_rotations(self):
        res = {}
        for name, limit in zip(['neck', 'jaw', 'global', 'eyes'],
                               [self.neck_limits, self.jaw_limits,
                                self.global_limits, self.eye_limits]):
            r_min, r_max = limit[:, 0], limit[:, 1]
            diff = r_max - r_min
            res[name] = torch.atanh(-2 * r_min / diff - 1)
            # assert (r_min + (torch.tanh(res[name]) + 1) / 2 * diff) < 1e-7
        return res

    def forward(self,
                shape,
                expr,
                rotation,
                neck,
                jaw,
                eyes,
                offsets=None,
                translation=None,
                zero_centered=True,
                use_rotation_limits=True,
                return_landmarks='static',
                return_joints=False,
                return_mouth_conditioning=False,
                **kwargs):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        device = self._v_template.device
        dtype = self._v_template.dtype
        N = len(shape)
        batch_size = shape.shape[0]

        # apply limits to joint rotations
        if use_rotation_limits:
            neck, jaw = self.apply_rotation_limits(neck=neck, jaw=jaw)

            if eyes is None:
                eyes = self.eye_pose.expand(batch_size, -1)
            else:
                eyes = torch.cat([self._apply_rotation_limit(eyes[:, :3], self.eye_limits),
                                  self._apply_rotation_limit(eyes[:, 3:], self.eye_limits)], dim=1)

        betas = torch.cat([shape, expr], dim=1)
        full_pose = torch.cat([rotation, neck, jaw, eyes], dim=1)
        template_vertices = self._v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # fill up offsets for filtered out vertices with 0s
        if offsets is not None and len(self._ignore_faces) != 0:
            offsets_ = torch.zeros(N, len(self._v_template), 3, dtype=dtype, device=device)
            offsets_.permute(1, 0, 2)[self._vert_filter] = offsets.permute(1, 0, 2)
        else:
            offsets_ = offsets

        faces = self._faces.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, J, mat_rot = lbs(betas, full_pose, offsets_, template_vertices,
                                   self.shapedirs, self.posedirs,
                                   self.J_regressor, self.parents,
                                   self.lbs_weights, dtype=self.dtype, faces=faces)

        if zero_centered:
            vertices = vertices - J[:, [0]]
            J = J - J[:, [0]]

        if translation is not None:
            vertices = vertices + translation[:, None, :]
            J = J + translation[:, None, :]

        filtered_vertices = vertices
        if len(self._ignore_faces) != 0:
            filtered_vertices = vertices.permute(1, 0, 2)[self._vert_filter].permute(1, 0, 2)

        ret_vals = [filtered_vertices]

        # compute landmarks if desired
        if return_landmarks is not None:
            assert return_landmarks in ['static', 'dynamic']
            if return_landmarks == 'static':
                bz = vertices.shape[0]
                landmarks = vertices2landmarks(vertices, self._faces,
                                               self.full_lmk_faces_idx.repeat(bz, 1),
                                               self.full_lmk_bary_coords.repeat(bz, 1, 1))
            else:
                lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
                lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

                dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
                    full_pose, self.dynamic_lmk_faces_idx,
                    self.dynamic_lmk_bary_coords,
                    self.neck_kin_chain, dtype=self.dtype)
                lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
                lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

                landmarks = vertices2landmarks(vertices, self._faces,
                                               lmk_faces_idx,
                                               lmk_bary_coords)

            ret_vals.append(landmarks)

        if return_joints:
            ret_vals.append(J)

        if return_mouth_conditioning:
            mouth_vert_pair_idcs = torch.tensor([[3506, 3531], [1670, 2906], [1738, 2882],
                                                 [1730, 2845], [1775, 2853], [1803, 2787],
                                                 [2853, 2882], [2787, 2906], [1670, 1803], [1738, 1775]])
            mouth_vert_coords = vertices.permute(1, 0, 2)[mouth_vert_pair_idcs]  # 6 x 2 x N x 3
            mouth_conditioning = (mouth_vert_coords[:, 0] -
                                  mouth_vert_coords[:, 1]).pow(2).sum(dim=-1).sqrt().permute(1, 0)  # N x 6

            # normalize mouth distances to range -1 ... 1 based on values obtained from philips sequence
            mins = torch.tensor([[0.0072, 0.0238, 0.0338, 0.0409, 0.0344, 0.0221, 0.0037, 0.0070, 0.0069, 0.0033]],
                                device=mouth_conditioning.device)
            maxs = torch.tensor([[0.0237, 0.0368, 0.0500, 0.0631, 0.0508, 0.0358, 0.0157, 0.0239, 0.0229, 0.0157]],
                                device=mouth_conditioning.device)
            mouth_conditioning = (mouth_conditioning - mins) / (maxs - mins) * 2 - 1

            # calculate nose rotation:
            rot_global = Rotation.from_rotvec(full_pose[:, :3].detach().cpu().numpy())
            rot_neck = Rotation.from_rotvec(full_pose[:, 3:6].detach().cpu().numpy())
            rot_nose = rot_global * rot_neck
            nose_condition = torch.tensor(rot_nose.as_rotvec(), device=mouth_conditioning.device,
                                          dtype=mouth_conditioning.dtype)
            mouth_conditioning = torch.cat((mouth_conditioning, nose_condition), dim=1)

            ret_vals.append(mouth_conditioning)

        if len(ret_vals) > 1:
            return ret_vals
        else:
            return ret_vals[0]

