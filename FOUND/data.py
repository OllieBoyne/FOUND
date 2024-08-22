from torch.utils.data import Dataset
import numpy as np
from utils.normal import kappa_to_alpha_np
from utils.colmap import load_colmap_data
from pytorch3d.renderer.cameras import get_world_to_view_transform
import os
import cv2
import warnings
import json
import torch

VALID_EXTS = ["png", "jpg", "jpeg"]


def _remove_ext(f):
    return os.path.splitext(f)[0]


class Cacher:
    """Cacher object to increase dataset loading speed."""

    def __init__(self, max_items=1000):
        """Store the most recent images in memory."""

        self.max_items = max_items
        self.cache = {}
        self.order = []

    def add(self, idx, sample):
        """Add item to cache."""
        self.cache[idx] = sample
        self.order.append(idx)
        if len(self.cache) > self.max_items:
            f = self.order.pop(0)
            del self.cache[f]


class FootScanDataset(Dataset):
    """Load a multiview captured foot scan dataset."""

    def __init__(self, src, targ_img_size, folder_names: dict):
        """

        :param src:
        :param targ_img_size: (H x W) Target image size
        :param folder_names: Dictionary of ftype: folder name. Will validate these while loading dataset
        """
        self.src = src
        self.idxs = []
        self.targ_img_size = targ_img_size

        for n, fname in folder_names.items():
            if not os.path.isdir(os.path.join(src, fname)):
                raise FileNotFoundError(f"Folder {fname} not found in {src}")

        self.rgb_dir = folder_names["rgb"]
        self.norm_dir = folder_names["norm"]
        self.norm_unc_dir = folder_names["norm_unc"]

        # get filenames of all rgb
        for f in os.listdir(os.path.join(src, self.rgb_dir)):
            self.idxs.append(_remove_ext(f))

        # load colmap data
        colmap_loc = os.path.join(src, "colmap.json")
        if os.path.isfile(colmap_loc):
            self.colmap_data = load_colmap_data(colmap_loc)
        else:
            raise FileNotFoundError(f"Colmap data not found at {colmap_loc}")

        # get image height to work out scaling factor
        # NOTE: assumes all images have same height
        loaded_img = self.load_img("rgb", self.idxs[0])
        self.resize_fac = fac = targ_img_size[0] / loaded_img.shape[0]

        f, cx, cy = [self.colmap_data["params"][i] for i in ["f", "cx", "cy"]]
        self.camera_params = dict(
            focal_length=f * fac, principal_point=(cx * fac, cy * fac)
        )

        # load GT Mesh
        # TODO

        # load keypoint labels
        kp_loc = os.path.join(src, "keypoints.json")
        if os.path.isfile(kp_loc):
            with open(kp_loc, "r") as f:
                kp_data = json.load(f)

            self.kp_labels = kp_data["kp_labels"]
        else:
            warnings.warn(f"Keypoint labels not found at {kp_loc}")
            self.kp_labels = None

        self.kp_data = {_remove_ext(k): v for k, v in kp_data["annotations"].items()}

        # load cacher
        self.cacher = Cacher()

    def restrict_views(self, n_views: int):
        """Sample n cameras uniformly across Y direction (avoiding repeats).
        Set this to be the new dataset"""

        # build R and T from data
        R = np.stack([self.colmap_data["R"][i] for i in self.idxs])
        T = np.stack([self.colmap_data["T"][i] for i in self.idxs])

        num_starting_views = len(self)

        w2v_trans = get_world_to_view_transform(
            R=torch.from_numpy(R), T=torch.from_numpy(T)
        )
        C = (
            w2v_trans.inverse().get_matrix()[:, 3, :3].cpu().detach().numpy()
        )  # camera centres

        cam_idxs = np.arange(num_starting_views)
        cam_idxs = sorted(cam_idxs, key=lambda i: C[i, 1])  # sort according to Y value

        if n_views == 1:
            cam_idxs = [int(np.median(cam_idxs))]  # middle view

        elif n_views == 2:
            cam_idxs = [cam_idxs[0], cam_idxs[-1]]  # start and end view

        else:
            # want to select a subset of cam_idxs which maximises the Y-wise distance between each camera
            # Simple algorithm:
            # Reject the camera closest in Y-value to its neighbours (starting from left)
            # Keep the leftmost & rightmost views always
            # Repeat until n_views cameras left
            while len(cam_idxs) > n_views:
                Ys = C[cam_idxs, 1]  # Y values
                dists = [
                    (abs(a - b) + abs(c - b)) / 2 for a, b, c in zip(Ys, Ys[1:], Ys[2:])
                ]  # middle cameras

                closest_cam = np.argmin(dists) + 1
                cam_idxs.pop(closest_cam)  # remove closest camera from list

        self.idxs = [self.idxs[i] for i in cam_idxs]

    def load_img(
        self, directory: str, loc: str, targ_size=None, raw=False
    ) -> np.ndarray:
        """
        Loads first image found with any of valid filetypes.
        Loads as float, [0 - 1]
        :param directory: Directory of file to load (within self.src)
        :param loc: filename (not including extension) to load
        :param targ_size: (W x H) Target image size
        :return:
        """

        for e in VALID_EXTS:
            pth = os.path.join(self.src, directory, f"{loc}.{e}")
            if os.path.isfile(pth):
                if raw:
                    rgb = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
                else:
                    rgb = cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2RGB)

                if targ_size != None:
                    current_aspect_ratio = rgb.shape[1] / rgb.shape[0]
                    targ_aspect_ratio = targ_size[1] / targ_size[0]
                    if current_aspect_ratio != targ_aspect_ratio:
                        raise ValueError(
                            f"Image {loc} has aspect ratio {current_aspect_ratio}, but target aspect ratio is {targ_aspect_ratio}."
                        )

                    rgb = cv2.resize(rgb, targ_size[::-1])

                return rgb.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        idx = self.idxs[i]

        if idx in self.cacher.cache:
            return self.cacher.cache[idx]

        rgb = self.load_img(self.rgb_dir, idx, self.targ_img_size)
        norm_rgb = self.load_img(self.norm_dir, idx, self.targ_img_size)
        norm_kappa = self.load_img(self.norm_unc_dir, idx, self.targ_img_size, raw=True)

        norm_xyz = norm_rgb * 2 - 1
        norm_alpha = kappa_to_alpha_np(norm_kappa)

        # load colmap
        R = self.colmap_data["R"][idx]
        T = self.colmap_data["T"][idx]

        # load keypoints
        kp_data = self.kp_data[idx]
        kps_raw = np.array(kp_data["kps"]) * self.resize_fac
        kps_vis = np.array(kp_data["vis"])
        kps_var = np.array(kp_data["variance"])

        kps = np.concatenate(
            [kps_raw, kps_vis[..., None]], axis=-1
        )  # resized (x, y) coord + vis flag [size K x 3]
        kps_unc = (
            kps_var * self.resize_fac**2
        )  # resized (sigma_x, sigma_y) ** 2 uncertainties [size K x 2]

        out = {
            "key": idx,
            "rgb": rgb,
            "norm_rgb": norm_rgb,
            "norm_xyz": norm_xyz,
            "norm_kappa": norm_kappa,
            "norm_alpha": norm_alpha,
            "R": R,
            "T": T,
            "kps": kps,
            "kps_unc": kps_unc,
        }

        self.cacher.add(idx, out)

        return out
