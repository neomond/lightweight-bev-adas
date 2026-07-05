"""nuScenes Dataset Loader for BEV Perception."""

import os, numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
DETECTION_CLASSES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]


class NuScenesDataset(Dataset):
    def __init__(
        self,
        dataroot,
        version="v1.0-mini",
        split="train",
        image_size=(640, 384),
        x_range=(-50.0, 50.0),
        y_range=(-50.0, 50.0),
        z_range=(-5.0, 3.0),
        max_lidar_points=35000,
    ):
        self.dataroot, self.image_size = dataroot, image_size
        self.x_range, self.y_range, self.z_range = x_range, y_range, z_range
        self.max_lidar_points = max_lidar_points
        print(f"Loading nuScenes {version} from {dataroot}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.samples = self._get_samples(split)
        print(f"Loaded {len(self.samples)} samples ({split} split)")

    def _get_samples(self, split):
        scenes = self.nusc.scene
        if "mini" in self.nusc.version:
            scene_names = [
                s["name"] for s in (scenes[:8] if split == "train" else scenes[8:])
            ]
        else:
            from nuscenes.utils.splits import create_splits_scenes

            scene_names = create_splits_scenes().get(split, [])
        samples = []
        for scene in self.nusc.scene:
            if scene["name"] in scene_names:
                token = scene["first_sample_token"]
                while token:
                    samples.append(token)
                    token = self.nusc.get("sample", token)["next"]
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.nusc.get("sample", self.samples[idx])
        camera_images, calibration = self._load_cameras(sample)
        lidar_points = self._load_lidar(sample)
        annotations = self._load_annotations(sample)
        return {
            "camera_images": camera_images,
            "lidar_points": lidar_points,
            "annotations": annotations,
            "calibration": calibration,
            "sample_token": self.samples[idx],
        }

    def _load_cameras(self, sample):
        images, calibration = [], {}
        for cam in CAMERA_CHANNELS:
            cam_data = self.nusc.get("sample_data", sample["data"][cam])
            img = Image.open(os.path.join(self.dataroot, cam_data["filename"])).convert(
                "RGB"
            )
            img = img.resize(self.image_size, Image.BILINEAR)
            img_tensor = torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)
            images.append(img_tensor)
            calib = self.nusc.get(
                "calibrated_sensor", cam_data["calibrated_sensor_token"]
            )
            ego = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
            calibration[cam] = {
                "intrinsic": torch.tensor(
                    calib["camera_intrinsic"], dtype=torch.float32
                ),
                "rotation": torch.tensor(
                    Quaternion(calib["rotation"]).rotation_matrix, dtype=torch.float32
                ),
                "translation": torch.tensor(calib["translation"], dtype=torch.float32),
                "ego_rotation": torch.tensor(
                    Quaternion(ego["rotation"]).rotation_matrix, dtype=torch.float32
                ),
                "ego_translation": torch.tensor(
                    ego["translation"], dtype=torch.float32
                ),
            }
        return torch.stack(images, dim=0), calibration

    def _load_lidar(self, sample):
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pc = LidarPointCloud.from_file(
            os.path.join(self.dataroot, lidar_data["filename"])
        )
        points = pc.points.T.astype(np.float32)  # (N, 4) — ensure float32 before matmul
        calib = self.nusc.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"]
        )
        rot = Quaternion(calib["rotation"]).rotation_matrix.astype(np.float32)
        trans = np.array(calib["translation"], dtype=np.float32)
        # Filter out sensor artifact points BEFORE transforming.
        # Raw .pcd.bin files contain a small number of points with extreme or
        # NaN/Inf values that cause float32 overflow in matmul. 200m is safely
        # beyond the 100m BEV range so no valid points are lost.
        pre_mask = (
            np.isfinite(points[:, :3]).all(axis=1) &
            (np.abs(points[:, 0]) < 200.0) &
            (np.abs(points[:, 1]) < 200.0) &
            (np.abs(points[:, 2]) < 200.0)
        )
        points = points[pre_mask]
        points[:, :3] = points[:, :3] @ rot.T + trans
        mask = (
            (points[:, 0] >= self.x_range[0])
            & (points[:, 0] < self.x_range[1])
            & (points[:, 1] >= self.y_range[0])
            & (points[:, 1] < self.y_range[1])
            & (points[:, 2] >= self.z_range[0])
            & (points[:, 2] < self.z_range[1])
        )
        points = points[mask]
        n = points.shape[0]
        if n > self.max_lidar_points:
            points = points[np.random.choice(n, self.max_lidar_points, replace=False)]
        elif n < self.max_lidar_points:
            points = np.vstack(
                [points, np.zeros((self.max_lidar_points - n, 4), dtype=np.float32)]
            )
        return torch.from_numpy(points.astype(np.float32))

    def _load_annotations(self, sample):
        boxes, classes, names = [], [], []
        cat_map = {
            "vehicle.car": "car",
            "vehicle.truck": "truck",
            "vehicle.bus.bendy": "bus",
            "vehicle.bus.rigid": "bus",
            "vehicle.trailer": "trailer",
            "vehicle.construction": "construction_vehicle",
            "human.pedestrian.adult": "pedestrian",
            "human.pedestrian.child": "pedestrian",
            "human.pedestrian.construction_worker": "pedestrian",
            "human.pedestrian.police_officer": "pedestrian",
            "vehicle.motorcycle": "motorcycle",
            "vehicle.bicycle": "bicycle",
            "movable_object.trafficcone": "traffic_cone",
            "movable_object.barrier": "barrier",
        }

        # Get ego pose to transform global coords to ego-relative
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        ego_translation = np.array(ego_pose["translation"])
        ego_rotation = Quaternion(ego_pose["rotation"])

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            cls = cat_map.get(ann["category_name"])
            if cls is None:
                continue

            # Transform from global to ego frame
            global_pos = np.array(ann["translation"])
            ego_pos = ego_rotation.inverse.rotate(global_pos - ego_translation)
            x, y, z = ego_pos

            w, l, h = ann["size"]
            global_yaw = Quaternion(ann["rotation"]).yaw_pitch_roll[0]
            ego_yaw_offset = ego_rotation.yaw_pitch_roll[0]
            yaw = global_yaw - ego_yaw_offset

            # Filter by range (now in ego-relative coords)
            if abs(x) > self.x_range[1] or abs(y) > self.y_range[1]:
                continue

            boxes.append([x, y, z, w, l, h, yaw])
            classes.append(DETECTION_CLASSES.index(cls))
            names.append(cls)

        if not boxes:
            return {
                "boxes": torch.zeros((0, 7)),
                "classes": torch.zeros((0,), dtype=torch.long),
                "names": [],
            }
        return {
            "boxes": torch.tensor(boxes),
            "classes": torch.tensor(classes, dtype=torch.long),
            "names": names,
        }


def collate_fn(batch):
    return {
        "camera_images": torch.stack([b["camera_images"] for b in batch]),
        "lidar_points": torch.stack([b["lidar_points"] for b in batch]),
        "annotations": [b["annotations"] for b in batch],
        "calibration": [b["calibration"] for b in batch],
        "sample_tokens": [b["sample_token"] for b in batch],
    }