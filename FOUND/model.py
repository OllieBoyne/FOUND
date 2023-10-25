import numpy as np
import torch
import trimesh
from functools import reduce
import json
import csv
import os

from utils.FIND.src.model.model import model_from_opts, process_opts

from pytorch3d import ops as p3d_ops
from pytorch3d.structures import Meshes

nn = torch.nn
class FIND(nn.Module):
	def __init__(self, find_dir: str, opt_posevec: bool=True, kp_labels: list=None):
		"""
		:param find_dir: directory containing the necessary files to build the FIND model
		:param optimize_posevec: Boolean flag for whether to optimise over the pose vector
		:param kp_labels: list of labels for the keypoints to extract from the mesh
		"""

		super().__init__()

		# check all necessary paths are here and load
		opts_loc = os.path.join(find_dir, 'opts.yaml')
		model_loc = os.path.join(find_dir, 'model.pth')
		keypoints_loc = os.path.join(find_dir, 'keypoints.csv')
		masked_faces_loc = os.path.join(find_dir, 'templ_masked_faces.npy')

		for f in [opts_loc, model_loc, keypoints_loc, masked_faces_loc]:
			if not os.path.isfile(f):
				raise FileNotFoundError(f"Could not find {f} in {find_dir}")


		opts = process_opts(opts_loc)
		opts.load_model = model_loc
		self.model = model_from_opts(opts)

		# define parameters
		self.rot = nn.Parameter(torch.tensor([0., 0., 0.], requires_grad=True).unsqueeze(0))
		self.trans = nn.Parameter(torch.tensor([0., 0., 0.], requires_grad=True).unsqueeze(0))
		self.scale = nn.Parameter(torch.tensor([1., 1., 1.], requires_grad=True).unsqueeze(0))

		vec_size = 100 # needs customisation
		self.shapevec = nn.Parameter \
			(torch.tensor([0. ] *vec_size, requires_grad=True).unsqueeze(0))
		self.posevec = nn.Parameter \
			(torch.tensor([0. ] *vec_size, requires_grad=True).unsqueeze(0))
		self.texvec = nn.Parameter \
			(torch.tensor([0. ] *vec_size, requires_grad=True).unsqueeze(0))


		self.params['reg'] = [self.rot, self.trans, self.scale]
		self.params['deform'] = [self.shapevec]
		self.params['vertex_deform'] = [self.surface_deforms]

		if opt_posevec:
			self.params['deform'] += [self.posevec]

		self.kps_idxs = None
		self.kp_labels = kp_labels
		if keypoints_loc is not None:
			with open(keypoints_loc) as infile:
				reader = csv.reader(infile)
				header = next(reader)
				for i in reader:
					if i[0] == 'FIND':
						if kp_labels is None: # get all keypoints of mesh
							self.kps_idxs = [*map(int, i[1:])]
							self.kp_labels = header[1:]

						else: # get keypoints corresponding to kp_labels
							self.kps_idxs = [int(i[header.index(l)]) for l in kp_labels]

		self.masked_face_idxs = None
		if masked_faces_loc is not None:
			self.masked_face_idxs = torch.from_numpy(np.load(masked_faces_loc))

	def __call__(self, ):

		reg = torch.cat([self.trans, self.rot, self.scale], dim=-1)

		meshes = self.model.get_meshes(shapevec=self.shapevec, reg=reg,
									   texvec=self.texvec, posevec=self.posevec)['meshes']

		# add surface deform
		if self.surface_deforms is not None:
			x = torch.tanh(self.surface_deforms) * self.max_surface_deform
			meshes = meshes.offset_verts(x.unsqueeze(-1) * meshes.verts_normals_packed())

		return meshes


	def kps_from_mesh(self, mesh: Meshes):
		return mesh.verts_padded()[:, self.kps_idxs]

	def get_params(self, keys):
		"""Given a list of param groups, return as single list"""
		return reduce(lambda a,b: a+b, [self.params[k] for k in keys])
	def get_mask_out_faces(self):
		"""Return list of faces to mask out"""
		return self.masked_face_idxs

	def save(self, loc):
		"""Save current parameters to output loc (as json)"""
		out = dict(reg = {}, deform = {}, vertex_deform = {})
		for name, param in self.named_parameters():
			for k in ['reg', 'deform', 'vertex_deform']:
				if any(param is x for x in self.params[k]):
					# only works with 1D
					out[k][name] = [f'{x:.4f}' for x in np.ravel(param.cpu().detach().numpy())]

		with open(loc, 'w') as outfile:
			json.dump(out, outfile)