from torch.utils.data import Dataset
import numpy as np
from utils.normal import kappa_to_alpha_np
import os
import cv2
import warnings

VALID_EXTS = ['.png', '.jpg', '.jpeg']

#TODO write cacher

class FootScanDataset(Dataset):
	"""Load a multiview captured foot scan dataset."""
	def __init__(self, src, targ_img_size, folder_names: dict):
		"""

		:param src:
		:param targ_img_size: (W x H) Target image size
		:param folder_names: Dictionary of ftype: folder name. Will validate these while loading dataset
		"""
		self.src = src
		self.idxs = []
		self.targ_img_size = targ_img_size

		for n, fname in folder_names.items():
			if not os.path.isdir(os.path.join(src, fname)):
				raise FileNotFoundError(f"Folder {fname} not found in {src}")

		# get filenames of all rgb
		for f in os.listdir(os.path.join(src, folder_names['rgb'])):
			self.idxs.append(os.path.splitext(f)[0])


		# load colmap
		self.camera_params = dict(focal_length = 0, principal_points= (0, 0,)) # TODO

		# load GT Mesh
		# TODO

		# load keypoint labels
		self.keypoint_labels = [] # TODO

	def load_img(self, directory: str, loc: str, targ_size = None) -> np.ndarray:
		"""
		Loads first image found with any of valid filetypes.
		Loads as float, [0 - 1]
		:param directory: Directory of file to load (within self.src)
		:param loc: filename (not including extension) to load
		:param targ_size: (W x H) Target image size
		:return:
		"""

		for e in VALID_EXTS:
			pth = os.path.join(self.src, directory, f'{loc}.{e}')
			if os.path.isfile(pth):
				rgb = cv2.cvtColor(cv2.imread(loc), cv2.COLOR_BGR2RGB)
				if targ_size != None:

					current_apsect_ratio = rgb.shape[1] / rgb.shape[0]
					targ_aspect_ratio = targ_size[1] / targ_size[0]
					if current_apsect_ratio != targ_aspect_ratio:
						warnings.warn(f"Image {loc} has aspect ratio {current_apsect_ratio}, but target aspect ratio is {targ_aspect_ratio}.")

					rgb = cv2.resize(rgb, targ_size)


				return rgb.astype(np.float32) / 255.0

	def __len__(self):
		return len(self.idxs)

	def __getitem__(self, i):
		idx = self.idxs[i]

		rgb = self.load_img('rgb', idx, self.targ_img_size)
		norm_rgb = self.load_img('norm', idx, self.targ_img_size)
		norm_kappa = self.load_img('norm_kappa', idx, self.targ_img_size)

		norm_xyz = norm_rgb * 2 - 1
		norm_alpha = kappa_to_alpha_np(norm_kappa)

		# load colmap

		# load keypoints

		return {
			'rgb': rgb,
			'norm_rgb': norm_rgb,
			'norm_xyz': norm_xyz,
			'norm_kappa': norm_kappa,
			'norm_alpha': norm_alpha,
		}