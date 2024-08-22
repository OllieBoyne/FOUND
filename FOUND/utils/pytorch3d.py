from pytorch3d.structures import Meshes
import trimesh
import torch

from pytorch3d.structures.utils import packed_to_list
from pytorch3d.renderer import TexturesVertex, TexturesUV
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures import Meshes
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments
from pytorch3d.ops.sample_points_from_meshes import _rand_barycentric_coords

import sys

F = torch.nn.functional

def extend_template(meshes: Meshes, N=1):
	"""Extend single mesh to multiple meshes """
	verts = meshes.verts_padded().expand(N, -1, -1).to(meshes.device)
	faces = meshes.faces_padded().expand(N, -1, -1).to(meshes.device)

	tex = None
	if meshes.textures is not None:
		tex = meshes.textures.expand(N, -1, -1)

	return Meshes(verts=verts, faces=faces, textures=tex)

def convert_to_textureVertex(textures_uv: TexturesUV, meshes:Meshes) -> TexturesVertex:
	verts_colors_packed = torch.zeros_like(meshes.verts_packed())
	verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()  # (*)
	return TexturesVertex(packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh()))


def to_trimesh(meshes: Meshes, idx=0, include_texture=True):
	"""Converts meshes to a single trimesh"""

	verts, faces = meshes.verts_padded().cpu().detach().numpy()[idx], meshes.faces_padded().cpu().detach().numpy()[idx]

	mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

	if include_texture:
		if isinstance(meshes.textures, TexturesVertex):
			mesh.visual = trimesh.visual.color.ColorVisuals(vertex_colors = meshes.textures.verts_features_padded().cpu().detach().numpy()[idx])
		elif isinstance(meshes.textures, TexturesUV):
			texture = convert_to_textureVertex(meshes.textures, meshes)
			mesh.visual = trimesh.visual.color.ColorVisuals(vertex_colors = texture.verts_features_padded().cpu().detach().numpy()[idx])
		else:
			raise NotImplementedError(f"Cannot export PyTorch3D Mesh to Trimesh with texture type {type(meshes.textures)}")

	return mesh

def export_mesh(mesh, export_loc, include_texture=True):
	"""Render displacement example AND template mesh"""

	mesh = to_trimesh(mesh, include_texture=True)
	obj_data = trimesh.exchange.obj.export_obj(mesh, include_color=True)
	with open(export_loc, 'w') as outfile:
		outfile.write(obj_data)


def modified_chamf(x,y, x_lengths=None, y_lengths=None,
    x_normals=None, y_normals=None,
    norm: int = 2):
    """
   	A modified version of pytorch3d.loss.chamfer_distance
   	to allow for no point or batch reduction and some other changes
    """

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0] ** .5  # (N, P1)
    cham_y = y_nn.dists[..., 0] ** .5  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0


    # Gather the normals using the indices and keep only value for k=0
    x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
    y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

    cham_norm_x = torch.abs(
        F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
    )
    cham_norm_y = torch.abs(
        F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
    )

    return dict(cham_x=cham_x, cham_y=cham_y, cham_norm_x = cham_norm_x, cham_norm_y=cham_norm_y)

def modified_sample(meshes: Meshes,  
    num_samples: int = 10000,
    return_normals: bool = False,
    return_textures: bool = False,):

    """Modified version of pytorch3d.ops.sample_points_from_meshes
    that returns references to the faces sampled from"""

    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    if return_textures and meshes.textures is None:
        raise ValueError("Meshes do not contain textures.")

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(
            areas, mesh_to_face[meshes.valid], max_faces
        )  # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        sample_face_idxs = areas_padded.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_valid_meshes, num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_face_idxs]
        normals[meshes.valid] = vert_normals

    if return_textures:
        # fragment data are of shape NxHxWxK. Here H=S, W=1 & K=1.
        pix_to_face = sample_face_idxs.view(len(meshes), num_samples, 1, 1)  # NxSx1x1
        bary = torch.stack((w0, w1, w2), dim=2).unsqueeze(2).unsqueeze(2)  # NxSx1x1x3
        # zbuf and dists are not used in `sample_textures` so we initialize them with dummy
        dummy = torch.zeros(
            (len(meshes), num_samples, 1, 1), device=meshes.device, dtype=torch.float32
        )  # NxSx1x1
        fragments = MeshFragments(
            pix_to_face=pix_to_face, zbuf=dummy, bary_coords=bary, dists=dummy
        )
        textures = meshes.sample_textures(fragments)  # NxSx1x1xC
        textures = textures[:, :, 0, 0, :]  # NxSxC

    out = {}

    out['verts'] = samples
    if return_normals: out['normals'] = normals
    if return_textures: out['textures'] = textures

    # return original faces
    out['face_idxs'] = sample_face_idxs

    return out