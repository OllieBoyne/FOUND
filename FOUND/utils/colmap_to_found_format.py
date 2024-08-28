"""Script for converting a COLMAP sparse reconstruction to FOUND format."""

import os
import argparse
from utils.colmap_read_write_model import read_model, qvec2rotmat
import numpy as np
import json

parser = argparse.ArgumentParser()

parser.add_argument('--colmap_sparse_dir', type=str, help="Path to the COLMAP sparse directory.")
parser.add_argument('--out_dir', type=str, help="Path to the output directory.")


def main(colmap_sparse_dir, out_dir):
	sparse_dir = colmap_sparse_dir

	os.makedirs(out_dir, exist_ok=True)

	# Get first available sparse fit.
	f = None
	for f in os.listdir(sparse_dir):
		if f.isdigit():
			f = os.path.join(sparse_dir, f)
			break
	else:
		raise FileNotFoundError(f"No model found for {colmap_sparse_dir}")

	cameras, images, points3D = read_model(path=f, ext='.bin')

	# From https://github.com/facebookresearch/pytorch3d/issues/1120
	# start by inverting X and Y to match with PyTorch3D coord system
	transf_c2p = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

	c = cameras[1]  # SIMPLE RADIAL CAMERA PARAMS: [f, cx, cy, k]
	camera_params = dict(zip(['width', 'height', 'f', 'cx', 'cy', 'k'], [c.width, c.height, *c.params]))

	output_data = dict(camera=camera_params)
	im_data = []
	for image_id, image in images.items():
		# build COLMAP matrix
		colmap_P = np.eye(4)
		colmap_P[:3, :3] = qvec2rotmat(image.qvec)
		colmap_P[:3, -1] = image.tvec

		# Calculate new centres
		original_cam_centres = -(qvec2rotmat(image.qvec).T @ image.tvec[:, None])[..., 0]

		# Calculate new rotations
		Transf_R = np.eye(3)
		R = (Transf_R @ qvec2rotmat(image.qvec).T).T

		# flip camera orientation from COLMAP to PyTorch3D
		R = transf_c2p @ R
		T = (- R @ original_cam_centres[:, None])[..., 0]  # Calculate new T

		entry = dict(image_id=image_id, pth=image.name, R=R.tolist(), C=original_cam_centres.tolist(), T=T.tolist())
		im_data.append(entry)

	# Export json
	output_data['images'] = im_data
	json_loc = os.path.join(out_dir, 'colmap.json')
	with open(json_loc, 'w') as outfile:
		json.dump(output_data, outfile)

	print(f'JSON saved to {json_loc}. Num images = {len(images)}')

if __name__ == "__main__":
	args = parser.parse_args()
	main(args.colmap_sparse_dir, args.out_dir)