from nha.util.log import get_logger
from nha.util.general import get_mask_bbox
from nha.util.render import create_intrinsics_matrix
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import cv2
import json
import numpy as np
import torch
import torchvision.transforms.functional as ttf
import pytorch_lightning as pl
import PIL.Image as Image
import argparse
import os

logger = get_logger(__name__)


def frame2id(frame_name):
    return int(frame_name.split("_")[-1])


def id2frame(frame_id):
    return f"frame_{frame_id:04d}"


def view2id(view_name):
    return int(view_name.split("_")[-1].split(".")[0])


def id2view(view_id):
    return f"image_{view_id:04d}.png"


SEGMENTATION_LABELS = {
    "body": 100,
    "head": 101,
    "hair": 102,
    "beard": 103,
    "clothes": 200,
    "head_wear": 201,
    "glasses": 202,
}

CLASS_IDCS = dict(
    background=0,
    skin=1,
    l_brow=2,
    r_brow=3,
    l_eye=4,
    r_eye=5,
    l_ear=7,
    r_ear=8,
    nose=10,
    mouth=11,
    u_lip=12,
    l_lip=13,
    neck=14,
    necklace=15,
    cloth=16,
    hair=17,
    headwear=18,
)


class RealDataset(Dataset):
    def __init__(
        self,
        path,
        frame_filter=None,
        tracking_results_path=None,
        tracking_resolution=None,
        load_uv=False,
        load_normal=False,
        load_lmk=False,
        lmk_dynamic=False,
        load_seg=False,
        load_parsing=False,
        load_bbx=False,
        load_flame=False,
        load_camera=False,
        load_light=False,
    ):
        """
        :param path: object of type Path from pathlib or str pointing
                     to the dataset root, which has the following
                     structure
                     -root/
                        |----subject1/
                        |        |----frame1/
                        |        ...
                        |----subject2/
                        ...
        :param per_view: If true a single view is considered a sample.
                         Otherwise, all views of a frame are considered
                         a sample.
        :param frame_filter: None or list which contains the frame
                             numbers to be considered
        :param load_uv: not implemented
        :param load_lmk: indicates if lmks shall be loaded
        :param lmk_dynamic: False: dynamic keypoints from face-alignment, True: static keypoints from openpose
        :param load_seg: load segmenation?
        :param load_bbx: not implemented
        :param load_flame: not implemented
        :param load_camera: indicates if camera parameters shall be loaded
        """
        super().__init__()
        self._path = Path(path)
        self._has_lmks = load_lmk
        self._has_flame = load_flame
        self._has_camera = load_camera
        self._lmks_dynamic = lmk_dynamic
        self._has_camera = load_camera
        self._has_seg = load_seg
        self._has_light = load_light
        self._has_normal = load_normal
        self._has_parsing = load_parsing

        self._eye_info_cache = {}

        # sanity checks for not implemented parts
        if load_uv:
            raise NotImplementedError("Real datasets don't contain uv-maps")
        if load_bbx:
            raise NotImplementedError("Real datasets don't contain bounding boxes")
        if load_flame:
            if tracking_results_path is None:
                raise ValueError(
                    "In order to load flame parameters, 'tracking_results_path' must be provided."
                )
        if load_camera:
            if tracking_results_path is None:
                raise ValueError(
                    "In order to load camera parameters, 'tracking_results_path' must be provided."
                )

        self._views = []

        self._load(frame_filter)
        self._tracking_results = (
            dict(np.load(tracking_results_path))
            if tracking_results_path is not None
            else None
        )
        self._tracking_resolution = tracking_resolution

        if self._tracking_resolution is None and self._tracking_results is not None:
            assert "image_size" in self._tracking_results
            self._tracking_resolution = self._tracking_results["image_size"]

        self._views = sorted(self._views, key=lambda x: frame2id(x.parent.name))

        self.max_frame_id = max([frame2id(x.parent.name) for x in self._views])

    def _load(self, frame_filter):
        """
        Loads the dataset from location self._path.
        The results are stored in the respective member
        variables.
        """

        # Obtain frame folders
        if frame_filter is None:
            frames = [f for f in os.listdir(self._path) if f.startswith("frame_")]
        else:
            frames = [f"frame_{f:04d}" for f in frame_filter]

        for f in frames:
            self._views.append(self._path / f / "image_0000.png")

        if (
            frame_filter is not None
        ):  # Attention: DOES NOT CHECK FRAMES FOR EACH SUBJECT INDIVIDUALLY!
            for f in frame_filter:
                if f not in self.frame_list:
                    raise FileNotFoundError(f"Couldn't find specified frame {f}")

    def __len__(self):
        if self._tracking_results is None:
            return len(self._views)
        else:
            return min(len(self._views), len(self._tracking_results["expr"]))

    def _get_eye_info(self, sample):
        if not sample["frame"] in self._eye_info_cache:
            eye_info = {}
            parsing = sample["parsing"][0].numpy()
            left_eye_parsing = parsing == CLASS_IDCS["l_eye"]
            right_eye_parsing = parsing == CLASS_IDCS["r_eye"]
            lmks = sample["lmk2d"].numpy().astype(int)

            right_eye = lmks[None, 36:42, :2]
            left_eye = lmks[None, 42:48, :2]

            left_eye_lmks = np.zeros(parsing.shape, dtype=np.uint8)
            right_eye_lmks = np.zeros(parsing.shape, dtype=np.uint8)
            cv2.fillPoly(right_eye_lmks, right_eye, 1)
            cv2.fillPoly(left_eye_lmks, left_eye, 1)

            is_good = []
            for l, p in zip(
                (left_eye_lmks, right_eye_lmks), (left_eye_parsing, right_eye_parsing)
            ):
                intersection = (l > 0) & (p > 0)
                union = (l > 0) | (p > 0)
                sum_union = np.sum(union)
                if sum_union == 0:
                    is_good.append(True)
                    continue

                iou = np.sum(intersection) / sum_union
                if iou > 0.5:
                    is_good.append(True)

            eye_info["valid_eyes"] = len(is_good) == 2

            left_bb = (
                get_mask_bbox(left_eye_parsing)
                if np.sum(left_eye_parsing) > 0
                else [0] * 4
            )
            right_bb = (
                get_mask_bbox(right_eye_parsing)
                if np.sum(right_eye_parsing) > 0
                else [0] * 4
            )
            eye_info["eye_distance"] = [
                left_bb[1] - left_bb[0],
                right_bb[1] - right_bb[0],
            ]

            self._eye_info_cache[sample["frame"]] = eye_info

        return self._eye_info_cache[sample["frame"]]

    def __getitem__(self, i):
        """
        Get i-th sample from the dataset.
        """

        view = self._views[i]
        frame_path = view.parent
        sample = {}

        # subject and frame info
        sample["frame"] = frame2id(frame_path.name)
        subject = frame_path.parent.name
        sample["subject"] = subject

        rgba = ttf.to_tensor(Image.open(view).convert("RGBA"))
        sample["rgb"] = (rgba[:3] - 0.5) / 0.5

        # segmentation
        if self._has_seg:
            seg_path = view.parent / view.name.replace("image", "seg")
            sample["seg"] = torch.from_numpy(np.array(Image.open(seg_path))).unsqueeze(
                0
            )

        # landmarks
        if self._has_lmks:
            path = frame_path / view.name.replace("image", "keypoints_static").replace(
                ".png", ".json"
            )
            if self._lmks_dynamic:
                path = frame_path / view.name.replace(
                    "image", "keypoints_dynamic"
                ).replace(".png", ".json")
            with open(path, "r") as f:
                lmks_info = json.load(f)
                lmks_view = lmks_info["people"][0]["face_keypoints_2d"]
                lmks_iris = lmks_info["people"][0].get("iris_keypoints_2d", None)
            # TODO: OpenPose has 70 facial landmarks, the synthetic dataset only 68

            sample["lmk2d"] = (
                torch.from_numpy(np.array(lmks_view)).float()[:204].view(-1, 3)
            )
            if lmks_iris is not None:
                sample["lmk2d_iris"] = torch.from_numpy(np.array(lmks_iris)).float()[
                    :204
                ]
                sample["lmk2d_iris"] = sample["lmk2d_iris"].view(-1, 3)[[1, 0]]

            if not self._lmks_dynamic:
                sample["lmk2d"][:, 2:] = 1.0
                if lmks_iris is not None:
                    if torch.sum(sample["lmk2d_iris"][:, :2] == -1) > 0:
                        sample["lmk2d_iris"][:, 2:] = 0.0
                    else:
                        sample["lmk2d_iris"][:, 2:] = 1.0

        # flame
        if self._has_flame:
            tr = self._tracking_results
            j = np.where(tr["frame"] == sample["frame"])[0][0]
            sample["flame_shape"] = torch.from_numpy(tr["shape"]).float()
            sample["flame_expr"] = torch.from_numpy(tr["expr"][j]).float()
            sample["flame_pose"] = torch.from_numpy(
                np.concatenate(
                    [
                        tr["rotation"][j],
                        tr["neck_pose"][j],
                        tr["jaw_pose"][j],
                        tr["eyes_pose"][j],
                    ],
                    axis=0,
                )
            ).float()
            sample["flame_trans"] = torch.from_numpy(tr["translation"][j]).float()

        # camera
        if self._has_camera:
            tr = self._tracking_results

            track_h, track_w = self._tracking_resolution
            img_h, img_w = sample["rgb"].shape[-2], sample["rgb"].shape[-1]

            fx_scale = max(track_h, track_w) * img_w / track_w
            fy_scale = max(track_h, track_w) * img_h / track_h
            cx_scale = img_w
            cy_scale = img_h
            if len(tr["K"].shape) == 1:
                sample["cam_intrinsic"] = create_intrinsics_matrix(
                    fx=tr["K"][0] * fx_scale,
                    fy=tr["K"][0] * fy_scale,
                    px=tr["K"][1] * cx_scale,
                    py=tr["K"][2] * cy_scale,
                )
            else:
                assert tr["K"].shape[0] == 3 and tr["K"].shape[1] == 3
                sample["cam_intrinsic"] = torch.from_numpy(tr["K"]).float()
            sample["cam_extrinsic"] = torch.from_numpy(tr["RT"]).float()

        if self._has_light:
            tr = self._tracking_results
            if len(tr["light"].shape) == 3:
                sample["light"] = torch.from_numpy(tr["light"][0]).float()
            else:
                sample["light"] = torch.from_numpy(tr["light"]).float()

        if self._has_normal:
            normal_path = view.parent / view.name.replace("image", "normals")
            normal = ttf.to_tensor(Image.open(normal_path).convert("RGB"))
            sample["normal"] = (normal - 0.5) / 0.5
            sample["normal"][1] *= -1

        if self._has_parsing:
            parsing_path = view.parent / view.name.replace("image", "parsing")
            parsing = torch.from_numpy(np.array(Image.open(parsing_path)))[None]
            sample["parsing"] = parsing
            sample.update(self._get_eye_info(sample))

        return sample

    @property
    def frame_list(self):
        frames = []
        for view in self._views:
            frames.append(frame2id(view.parent.name))
        return frames

    @property
    def view_list(self):
        views = []
        for view in self._views:
            views.append(view.name)
        return views


class RealDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        split_config=None,
        tracking_results_path=None,
        tracking_resolution=None,
        train_batch_size=64,
        validation_batch_size=64,
        data_worker=8,
        load_bbx=False,
        load_seg=False,
        load_flame=False,
        load_lmk=False,
        lmk_dynamic=False,
        load_uv=False,
        load_normal=False,
        load_camera=False,
        load_light=False,
        load_parsing=False,
        **kwargs,
    ):
        """
        Encapsulates train and validation splits of the real dataset and their dataloaders
        :param data_path: path to real dataset
        :param split_config: json file that specifies which frames to use for training and which for testing.
                            If None, all available frames are used for training and validation
        :param train_batch_size:
        :param validation_batch_size:
        :param loader_threads: number of workers to be spawned by the dataloaders
        :param load_cameras:
        :param load_flame:
        :param load_lmk:
        :param load_uv:
        :param load_normal:
        """
        super().__init__()
        self._path = Path(data_path)
        self._train_batch = train_batch_size
        self._val_batch = validation_batch_size
        self._workers = data_worker
        self._tracking_results_path = tracking_results_path
        self._tracking_resolution = tracking_resolution
        self._train_set = None
        self._val_set = None
        self._split_config = split_config
        self._load_components = dict(
            load_bbx=load_bbx,
            load_seg=load_seg,
            load_flame=load_flame,
            load_lmk=load_lmk,
            lmk_dynamic=lmk_dynamic,
            load_uv=load_uv,
            load_camera=load_camera,
            load_normal=load_normal,
            load_light=load_light,
            load_parsing=load_parsing,
        )

    def setup(self, stage=None):
        train_split, val_split = self._read_splits(self._split_config)

        self._train_set = RealDataset(
            self._path,
            frame_filter=train_split,
            tracking_results_path=self._tracking_results_path,
            tracking_resolution=self._tracking_resolution,
            **self._load_components,
        )

        logger.info(
            f"Collected real training dataset containing: "
            f"{len(self._train_set)} samples."
        )

        self._val_set = RealDataset(
            self._path,
            frame_filter=val_split,
            tracking_results_path=self._tracking_results_path,
            tracking_resolution=self._tracking_resolution,
            **self._load_components,
        )

        logger.info(
            f"Collected real validation dataset containing: "
            f"{len(self._val_set)} samples."
        )

    def swap_split(self):
        val_set = self._val_set
        train_set = self._train_set
        self._train_set = val_set
        self._val_set = train_set

    @property
    def max_frame_id(self):
        return max(self._train_set.max_frame_id, self._val_set.max_frame_id)

    @staticmethod
    def _read_splits(split_config):
        """
        Reads the train/val split information from the split file
        :param split_config:
        :return:
        """
        if split_config is None:
            return None, None

        else:
            with open(split_config, "r") as f:
                splits = json.load(f)
                train_split = splits["train"]
                val_split = splits["val"]

        return train_split, val_split

    @classmethod
    def add_argparse_args(cls, parser):
        """
        Adds dataset specific parameters to parser
        :param parser:
        :return:
        """
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument("--data_path", type=str, required=True)
        parser.add_argument("--data_worker", type=int, default=8)
        parser.add_argument("--split_config", type=str, required=False, default=None)
        parser.add_argument("--train_batch_size", type=int, default=8, nargs=3)
        parser.add_argument("--validation_batch_size", type=int, default=8, nargs=3)
        parser.add_argument("--tracking_results_path", type=Path, default=None)
        parser.add_argument("--tracking_resolution", type=int, default=None, nargs=2)
        parser.add_argument("--load_uv", action="store_true")
        parser.add_argument("--load_normal", action="store_true")
        parser.add_argument("--load_flame", action="store_true")
        parser.add_argument("--load_bbx", action="store_true")
        parser.add_argument("--load_lmk", action="store_true")
        parser.add_argument("--lmk_dynamic", action="store_true")
        parser.add_argument("--load_seg", action="store_true")
        parser.add_argument("--load_camera", action="store_true")
        parser.add_argument("--load_light", action="store_true")
        parser.add_argument("--load_parsing", action="store_true")

        return parser

    def train_dataloader(self, batch_size):
        return DataLoader(
            self._train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._workers,
        )

    def val_dataloader(self, batch_size):
        return DataLoader(
            self._val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._workers,
        )

    def prepare_data(self, *args, **kwargs):
        pass


def digitize_segmap(segmap, neglect=["clothes"]):
    """
    converts segmentation tensor with entries corresponding to the different classes to a binary foreground background
    mask depending on the specified values in neglect (list of keys of SEGMENTATION_LABELS dict)

    :param segmap:
    :param neglect_clothes:
    :return:
    """

    segmap = segmap.clone()
    for neglect_key in neglect:
        segmap[segmap == SEGMENTATION_LABELS[neglect_key]] = 0
    segmap[segmap != 0] = 1

    return segmap


def tracking_results_2_data_batch(tr: dict, idcs: list):
    """
    transforms tracking results entries to batch that can be processed be NHAOptimizer.forward()

    :param tr: tracking results
    :param idcs: frame indices
    :return:
    """
    # creating batch with inputs to avatar

    N = len(idcs)
    tr_idcs = np.array([np.where(tr["frame"] == i)[0][0] for i in idcs])

    img_h, img_w = tr['image_size']

    f_scale = max(img_h, img_w)
    cx_scale = img_w
    cy_scale = img_h

    cam_intrinsics = create_intrinsics_matrix(
        fx=tr["K"][0] * f_scale,
        fy=tr["K"][0] * f_scale,
        px=tr["K"][1] * cx_scale,
        py=tr["K"][2] * cy_scale,
    )


    pose = np.concatenate(
        [
            tr["rotation"],
            tr["neck_pose"],
            tr["jaw_pose"],
            tr["eyes_pose"],
        ],
        axis=1,
    )

    batch = dict(
        flame_shape=torch.from_numpy(tr["shape"][None]).float().expand(N, -1),
        flame_expr=torch.from_numpy(tr["expr"][tr_idcs]).float(),
        flame_pose=torch.from_numpy(pose[tr_idcs]).float(),
        flame_trans=torch.from_numpy(tr["translation"][tr_idcs]).float(),
        cam_intrinsic=cam_intrinsics[None].expand(N, -1, -1),
        cam_extrinsic=torch.from_numpy(tr["RT"]).float()[None].expand(N, -1, -1),
        frame=torch.tensor(idcs),
        rgb=torch.zeros(N, 3, img_h, img_w),
    )

    return batch
