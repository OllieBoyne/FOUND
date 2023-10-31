# Add FIND folders to path
import sys

sys.path.append('FIND')
sys.path.append('FIND/src/model')

import torch

from pytorch3d import ops as p3d_ops

import numpy as np

from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, FoVPerspectiveCameras, \
	look_at_view_transform, SoftPhongShader, SoftSilhouetteShader, PerspectiveCameras, AmbientLights, PointLights


class Normals:

	def __init__(self, xyz: torch.Tensor, mask: torch.Tensor = None, device=None):
		"""Takes a tensor of size [..., 3] of normals (do not need to be unit),
		processes them, and can transfer between different fmts, and convert to and from RGB.

		Must receive normals in PyTorch3D's XYZ co-ordinate system - can convert to different on 'rgb' call

		Optional mask to apply to all elements"""

		xyz = nn.functional.normalize(xyz, dim=-1)
		self._xyz = xyz

		self.mask = None
		if mask is not None:
			self.mask = mask.bool()
			assert self.mask.shape == xyz.shape[:-1], "Mask shape must match xyz in every dimension but last."

		if device is None:
			device = self._xyz.device

		self.to(device)

	def to_rgb(self, format='pytorch3d', mask_value=0.):

		normals = self.xyz.clone()

		if format == 'pytorch3d':
			# PyTorch3D: (XYZ) = (left, up, front)
			pass

		elif format == 'blender':
			# BLENDER: (XYZ) = (right, up, front)
			normals[..., 0] = -1 * normals[..., 0]

		elif format == 'ours':
			# Our format: (XYZ) = (left, up, back)
			normals[..., 2] = -1 * normals[..., 2]

		else:
			raise NotImplementedError(f"Normals format {format} not implemented.")

		rgb = (normals + 1) / 2  # normalize to lie in 0-1 range

		if self.mask is not None:
			rgb[~self.mask] = mask_value

		return rgb  # Normalize values to lie in range (0 - 1)

	def to_xyz(self, format='pytorch3d'):
		"""get normals (this part would probably be redundant
			I added it to get normals in different formats)
		"""
		normals = self.xyz.clone()

		if format == 'pytorch3d':
			pass
		elif format == 'blender':
			normals[..., 0] = -1 * normals[..., 0]
		elif format == 'ours':
			normals[..., 2] = -1 * normals[..., 2]
		else:
			raise NotImplementedError

		if self.mask is not None:
			normals[~self.mask] = 0
		return normals

	def mean(self):
		"""Return mean normal over all dimensions"""

		xyz = self.xyz
		if self.mask is not None:
			xyz = xyz[self.mask]

		xyz = xyz.reshape(-1, 3)
		return nn.functional.normalize(xyz.mean(dim=0), dim=-1)

	def to(self, device='cuda'):
		self._xyz = self._xyz.to(device)
		if self.mask is not None:
			self.mask = self.mask.to(device)

	@property
	def xyz(self):
		xyz = self._xyz
		if self.mask is not None:
			xyz = xyz * self.mask.unsqueeze(-1)
		return xyz

	def sample(self, points, padding_mode='zeros'):
		"""Given a grid of points to sample from [H, W, 2], sample these pixels in the xyz values of the normals.

		Return as [H x W x 3]"""
		return torch.nn.functional.grid_sample(self.xyz.permute(2, 1, 0).unsqueeze(0), points.unsqueeze(0),
											   padding_mode=padding_mode)[0].permute(1, 2, 0)


def normals_from_rgb(rgb: torch.Tensor, mask=None, format='pytorch3d') -> Normals:
	"""Given an RGB axis format, convert RGB normals into XYZ, return as a Normals object"""
	if mask is None:
		mask = torch.any(rgb > 0, dim=-1)

	xyz = (2 * rgb) - 1

	if format == 'pytorch3d':
		# PyTorch3D: (XYZ) = (left, up, front)
		pass

	elif format == 'blender':
		# BLENDER: (XYZ) = (right, up, front)
		xyz[..., 0] = -1 * xyz[..., 0]

	elif format == 'ours':
		# Gwanbgin: (XYZ) = (left, up, back)
		xyz[..., 2] = -1 * xyz[..., 2]

	else:
		raise NotImplementedError

	return Normals(xyz, mask)


def get_padded_pix_to_face(pix_to_face, meshes):
	# Need to convert pix_to_face to the face count in the specific mesh
	# By default, PyTorch3D counts the face indices cumulatively (eg if mesh 1 has F1 faces, the first face in mesh2 is)
	faces_per_mesh = meshes.num_faces_per_mesh()
	faces_cumulative = torch.cumsum(faces_per_mesh, dim=0) - faces_per_mesh
	pix_to_face = pix_to_face - faces_cumulative.unsqueeze(-1).unsqueeze(-1) * (pix_to_face >= 0)
	return pix_to_face


nn = torch.nn


class NormalShader(nn.Module):

	def forward(self, fragments, meshes, cameras) -> Normals:
		"""Return Normals object"""

		faces = meshes.faces_packed()  # (F, 3)
		vertex_normals = meshes.verts_normals_packed()  # (V, 3)
		faces_normals = vertex_normals[faces]
		pixel_normals = p3d_ops.interpolate_face_attributes(
			fragments.pix_to_face, fragments.bary_coords, faces_normals
		)

		pixel_normals = pixel_normals[:, :, :, 0]  # Dimension 3 is size faces_per_pixel. Take closest face to camera

		# Convert to camera reference frame
		P = cameras.get_world_to_view_transform()
		pshape = pixel_normals.shape
		Nviews = pshape[0]
		pixel_normals = P.transform_normals(pixel_normals.view(Nviews, -1, 3)).view(*pshape)
		mask = fragments.pix_to_face[..., 0] >= 0  # Mask out all pixels with no face

		return Normals(pixel_normals, mask=mask)


class Renderer(nn.Module):

	def __init__(self, device='cuda', image_size=(256, 256),
				 bin_size=None, z_clip_value=None,
				 max_faces_per_bin=None, cam_params: dict = None,
				 MAX_BATCH_SIZE=10,
				 **kwargs):

		super().__init__()

		self.MAX_BATCH_SIZE = MAX_BATCH_SIZE

		if isinstance(image_size, int):
			image_size = (image_size, image_size)

		self.image_size = image_size

		self.img_raster_settings = RasterizationSettings(
			image_size=image_size, blur_radius=0.,
			faces_per_pixel=1, max_faces_per_bin=max_faces_per_bin,
			bin_size=bin_size, z_clip_value=z_clip_value)

		# Rasterization settings for silhouette rendering
		sigma = 1e-6
		self.raster_settings_silhouette = RasterizationSettings(
			image_size=image_size,
			blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
			faces_per_pixel=10, max_faces_per_bin=max_faces_per_bin,
			bin_size=bin_size
		)

		self.rasterizer = MeshRasterizer(raster_settings=self.img_raster_settings)
		self.sil_rasterizer = MeshRasterizer(raster_settings=self.raster_settings_silhouette)

		# Shaders
		self.img_shader = SoftPhongShader(device=device)
		self.norm_shader = NormalShader()
		self.sil_shader = SoftSilhouetteShader()

		# default lighting
		self.lights = AmbientLights(device=device)

		self.camera_params = {}
		if cam_params is not None:
			# Multiple camera intrinsics not currently supported
			f = torch.tensor([[cam_params['focal_length']]]).to(device)  # [N x 1]
			pp = torch.tensor(cam_params['principal_point']).unsqueeze(0).to(device)  # [N x 2]
			self.camera_params = dict(focal_length=f, principal_point=pp,
									  in_ndc=False, image_size=torch.tensor(image_size).unsqueeze(0).to(device))

	def forward(self, meshes, R: torch.Tensor, T: torch.Tensor, keypoints=None,
				render_normals=True, render_rgb=True, render_sil=True,
				mask_out_faces=None, return_cameras=False, camera_params=None,
				normals_fmt='blender', one_view_per_mesh=False):
		"""
		Can receive various number of 'views' (size of R) and meshes (size of 'meshes')
		N input views, 1 mesh -> render N views of 1 mesh
		N input views, N mesh -> render one view per mesh (only if one_view_per_mesh is True)
		N input views, M mesh -> render N views of M meshes

		Render modes:
			- render_rgb: render RGB image
			- render_normals: render surface normals
			- render_sil: render silhouette
			- keypoints: project 3D keypoints onto image

		:param R: [N x 4 x 4]
		:param T: [N x 4 x 4]
		:param keypoints: optional [M x P x 3] keypoints to render
		:param mask_out_faces: [M x F] faces per mesh to optionally remove from seg & normal
		:param camera_params: Optional per-camera focal length & principal point
		:return:

		Currently does not support M > 1 rendering to M images.
		"""

		if camera_params is None:
			camera_params = self.camera_params

		N = R.shape[0]  # number of views
		M = len(meshes)  # number of meshes

		if M > 1 and (N == M):
			assert one_view_per_mesh, "For N == M, M > 1, requires one_view_per_mesh=True parameter."

			out_shape_rgb = (N, *self.image_size, 3)
			out_shape_single = (N, *self.image_size)
			batch_size = N

		# in the case M != N for M > 1, want to render all N views for each mesh
		elif M != N and M > 1:
			meshes = meshes.extend(N)  # produce a mesh for each view
			R = torch.cat([R] * M, dim=0)
			T = torch.cat([T] * M, dim=0)  # produce R, T for each mesh

			out_shape_rgb = (N, M, *self.image_size, 3)
			out_shape_single = (N, M, *self.image_size)
			batch_size = N * M

		# in the case M = 1, N >= 1, render N views of 1 mesh
		else:
			meshes = meshes.extend(N)  # produce a mesh for each view
			out_shape_rgb = (N, *self.image_size, 3)
			out_shape_single = (N, *self.image_size)

		cameras = PerspectiveCameras(device=meshes.device, R=R, T=T, **camera_params)

		out = dict()
		_frags = None
		normals = None
		if render_rgb or render_normals:
			fragments = self.rasterizer(meshes, cameras=cameras)
			_frags = fragments  # Store fragments for mask out faces

			if render_rgb:
				out['rgb'] = self.img_shader(fragments, meshes, cameras=cameras, lights=self.lights)[..., :3].reshape(
					out_shape_rgb)

			if render_normals:
				normals = self.norm_shader(fragments, meshes, cameras=cameras)

		if render_sil:
			fragments_sil = self.sil_rasterizer(meshes, cameras=cameras)
			if _frags is None: _frags = fragments_sil  # Store fragments for mask out faces

			sil = self.sil_shader(fragments_sil, meshes, cameras=cameras)
			out['sil'] = sil[..., -1].reshape(out_shape_single)  # return just alpha channel (silhouette)

		# Apply face masking of FIND model
		if (render_rgb or render_sil or render_normals) and mask_out_faces is not None:
			# get foremost face for each pixel in correct format
			pix_to_face = get_padded_pix_to_face(_frags.pix_to_face[..., 0], meshes).reshape(out_shape_single)

			for n in range(N):
				mask_pix = torch.isin(pix_to_face[n], mask_out_faces)

				if render_rgb:
					out['rgb'][n][mask_pix] = 1.  # set pixels to white

				if render_sil:
					out['sil'][n, mask_pix] = 0.

				if render_normals:
					normals.mask[n] *= ~mask_pix  # does not work for certain batch types

		if render_normals:
			# Also return rgb and xyz of normals
			out['norm_rgb'] = normals.to_rgb(format=normals_fmt, mask_value=.5).reshape(out_shape_rgb)
			out['norm_xyz'] = normals.to_xyz(format=normals_fmt).reshape(out_shape_rgb)

		if keypoints is not None:
			kps_2d = cameras.transform_points_screen(keypoints, image_size=self.image_size)[..., :2]
			out['kps'] = kps_2d

		if return_cameras:
			out['cameras'] = cameras

		return out


def view_from(view_kw='topdown', dist=.35):
	kws = ['topdown', 'side1', 'side2', 'toes', '45', '60']

	if isinstance(view_kw, str):
		view_kw = [view_kw]

	N = len(view_kw)
	R, T = torch.empty((N, 3, 3)), torch.empty((N, 3))
	for n, v in enumerate(view_kw):
		assert v in kws or isinstance(v, int), f"View description `{view_kw}` not understood"

		dist, elev, azim, point = dist, 0, 0, ((0, 0, 0),)
		if v == 'topdown': elev = 0
		if v == 'side1': elev = 90
		if v == 'side2': elev, azim = -90, 180
		if v == 'toes': point = ((0.1, 0, 0),); dist = 0.1
		if isinstance(v, int):
			elev = v

		_R, _T = look_at_view_transform(dist=dist, elev=elev, azim=azim, up=((1, 0, 0),), at=point)

		R[n] = _R
		T[n] = _T

	return R, T


def linspace_views(nviews=1, dist=.3, dist_min=None, dist_max=None,
				   elev_min=None, elev_max=None, azim_min=None, azim_max=None, at=((0, 0, 0),)):
	if dist_min is not None:
		dist = np.linspace(dist_min, dist_max, nviews)

	if elev_min is None:
		elev = 0
	else:
		elev = np.linspace(elev_min, elev_max, nviews)

	if azim_min is None:
		azim = 0
	else:
		azim = np.linspace(azim_min, azim_max, nviews)
	R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, up=((1, 0, 0),),
								  at=at)  # sample random viewing angles and distances
	return R, T