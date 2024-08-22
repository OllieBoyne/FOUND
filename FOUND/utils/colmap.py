import numpy as np
import json
import os


def _remove_ext(f):
    return os.path.splitext(f)[0]


def load_colmap_data(colmap_json: str, image_list: list = None):
    """Load R, t values from COLMAP for images,
    image_list = optional list N
    returns dict of R [ N x 3 x 3 ], t [ N x 3 ], params (camera params).

    Note that coordinates have already been converted into PyTorch3D's coord system in the data generation script.
    """

    with open(colmap_json) as infile:
        data = json.load(infile)

    if image_list is None:
        image_list = [x["pth"] for x in data["images"]]

    N = len(image_list)
    R = {}
    T = {}

    for n, i in enumerate(image_list):
        idx = [x for x in data["images"] if x["pth"] == i]
        if len(idx) == 0:
            raise ValueError(f"No COLMAP data found for {i}")
        elif len(idx) > 1:
            raise ValueError(f"{len(idx)} COLMAP data points found for {i}")

        image_data = idx[0]
        pth = _remove_ext(image_data["pth"])

        R[pth] = np.array(image_data["R"]).T
        T[pth] = np.array(image_data["T"])

    return dict(image_list=image_list, R=R, T=T, params=data["camera"])
