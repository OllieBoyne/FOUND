"""Evaluate the performance of a fitted mesh"""

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
import os
import trimesh
import cv2
from FIND.src.utils.pytorch3d_tools import to_trimesh
from find_norm.model.p3d_mod_fun import modified_chamf, modified_sample
from multiprocessing import Process
import multiprocessing as mp

from prettytable import PrettyTable
import torch
import torch.nn.functional as F

import numpy as np
from find_norm.renderer import Renderer, view_from
from find_norm.model.utils import produce_grid, put_text, colourbar
from matplotlib import pyplot as plt
import json

device = 'cuda'


def eval_metrics(arr, cutoffs=[5, 7.5, 11.25, 22.5, 30]):
	"""Given a 1d array, return mean, median, rmse,
	and % of values less than each in `cutoffs`"""
	assert arr.ndim == 1, "eval_metrics requires 1D array"

	out = dict(mean = arr.mean(), median = np.median(arr), rmse = (arr ** 2).mean() **.5,
			   cutoffs = [(arr < i).mean() for i in cutoffs])

	return out

def err_to_colour(err: torch.Tensor, vmin:float=None, vmax:float=None, colmin=(0, 1, 0), colmax=(1, 0, 0), nan_colour=(0.3, 0.3, 0.3)):
	"""Convert a tensor of errors (...) to an RGB colour scale (..., 3).
	 Linearly interpolate so that err of vmin -> colmin, err of vmax -> colmax
	 if vmin and vmax not given, take min and max of err
	 
	 If any nan's given, set their colour to nan_colour
	 """

	ndim = err.ndim
	colmin = torch.tensor(colmin)[(None,)*ndim].to(err.device) # expand colmin to [..., 3]
	colmax = torch.tensor(colmax)[(None,)*ndim].to(err.device)
	colnan = torch.tensor(nan_colour)[(None,)*ndim].to(err.device)

	vmin = err.nanmin() if vmin is None else vmin
	vmax = err.nanmax() if vmax is None else vmax

	fracs = (err - vmin) / (vmax - vmin)

	rgba = (colmin + fracs.unsqueeze(-1) * (colmax - colmin)).to(err.device)
	rgba = torch.clip(rgba, min=0, max=1)

	rgba[torch.any(torch.isnan(rgba), dim=-1)] = colnan

	return rgba

class Reporter:
	"""Receive statements, on exit print all and save all to file"""
	def __init__(self, out_file_loc):
		self.lines = []
		self.out_file_loc = out_file_loc

	def __call__(self, line):
		self.lines.append(line)
	
	def __enter__(self, *args):
		return self
	
	def __exit__(self, *args):
		[*map(print, self.lines)]
		with open(self.out_file_loc, 'w') as outfile:
			outfile.writelines([s + '\n' for s in self.lines])


def get_max_fit(exp_dir):
	"""Search in an experiment directory for the fit_xx.obj with the highest value"""
	f = lambda s: -1 if 'fit_' not in s else int(s.split('fit_')[1].split('.obj')[0]) 
	return max(os.listdir(exp_dir), key=f)

def cutoff_slice_FIND(mesh, max_heel_height = 0.04, cutoff_height = 0.1):
	"""Similar mesh slicing method to FIND: identify heel keypoint, slice off 1cm above"""
	X, Y, Z = mesh.vertices.T
	Xma = np.ma.array(X, mask= Z >= max_heel_height)
	heel_idx = np.ma.argmin(Xma)

	slice_height = min(Z[heel_idx] + cutoff_height, Z.max() - 5e-3)
	return mesh.slice_plane([0, 0, slice_height], [0, 0, -1], cap=False)

def get_loghist(x, nbins):
	hist, bins = np.histogram(x, bins=nbins)
	logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
	return dict(x=x, bins=logbins)

def eval_exp(exp_dir, render=True):
	results = {} # return results as errors

	if not any('fit_' in f for f in os.listdir(exp_dir)):
		print(f"No fits for {exp_dir}, skipping...")
		return
	
	pred_obj_loc = os.path.join(exp_dir, get_max_fit(exp_dir))

	# load settings to get folder
	opts_loc = os.path.join(exp_dir, 'opts.json')
	if not os.path.isfile(opts_loc):
		print(f"No opts for {exp_dir}, skipping...")
		return
	
	with open(opts_loc) as infile:
		settings = json.load(infile)
	
	# assume GT OBJ loc is
	# (1) saved in <data_folder>/mesh.obj if <data_folder> given
	if 'data_folder' in settings:
		gt_obj_loc = os.path.join(settings['data_folder'], 'mesh.obj')
	
	# (2) saved in <exp_dir>/gt_mesh.obj otherwise
	else:
		gt_obj_loc = os.path.join(exp_dir, 'gt_mesh.obj')

	eval_dir = os.path.join(exp_dir, 'eval')
	os.makedirs(eval_dir, exist_ok=True)

	with open(gt_obj_loc) as infile:
		d = trimesh.exchange.obj.load_obj(infile, process=False)
		gt_mesh_trimesh = trimesh.Trimesh(**d)

	with open(pred_obj_loc) as infile:
		d = trimesh.exchange.obj.load_obj(infile, process=False)
		pred_mesh_trimesh = trimesh.Trimesh(**d)

	# pre-process meshes, w/ cutoff
	# Same method as used for Foot3D here for slicing GT
	gt_mesh_trimesh = cutoff_slice_FIND(gt_mesh_trimesh)

	if settings.get('model', 'FIND') == 'FIND':
		# slice FIND faces
		FIND_cutoff_surface = np.load(os.path.join(settings['find_pth'], 'templ_masked_faces.npy'))
		FIND_sole_faces = np.load(os.path.join(settings['find_pth'], 'templ_sole_faces.npy'))
		FIND_sole_verts = np.unique(np.ravel(pred_mesh_trimesh.faces[FIND_sole_faces])) # all vertices considered part of the sole

		sole_vert_positions = pred_mesh_trimesh.vertices[FIND_sole_verts] # save sole vertex positions to refind them after mesh pre-processing

		pred_mesh_trimesh.update_faces(~np.isin(np.arange(pred_mesh_trimesh.faces.shape[0]), FIND_cutoff_surface))
		pred_mesh_trimesh = cutoff_slice_FIND(pred_mesh_trimesh)
		
		# define a mask
		# want to be able to define a mask on the FIND model, so that errors of verts in this mask aren't considered real -> pred, but are considered in reverse
		# (for sole verts, unfair to count the error on them, but likewise incorrect to just remove them all, especially at the boundary)
		# recalculate sole vertices
		FIND_sole_vert_idxs = np.argwhere(np.all(pred_mesh_trimesh.vertices[:, None, :] == sole_vert_positions[None, ...], axis=-1))[:, 0]
		
		FIND_sole_vertex_mask = np.isin(np.arange(pred_mesh_trimesh.vertices.shape[0]), FIND_sole_vert_idxs) # mask of which vertices correspond to the sole
		FIND_sole_faces_mask = np.any(FIND_sole_vertex_mask[pred_mesh_trimesh.faces], axis=-1) # mask of which faces are in sole

	else:
		pred_mesh_trimesh = cutoff_slice_FIND(pred_mesh_trimesh)

	# Convert to PyTorch3D
	p3d_from_trimesh = lambda mesh: Meshes(verts=torch.from_numpy(np.asarray(mesh.vertices)[None, ...]).float(),
										   faces=torch.from_numpy(np.asarray(mesh.faces)[None, ...])).to(device)

	gt_mesh = p3d_from_trimesh(gt_mesh_trimesh)
	pred_mesh = p3d_from_trimesh(pred_mesh_trimesh)

	# Sample vertices uniformly from mesh, returning vertex position, normal, and original face/vert idxs
	gt_sample_dict = modified_sample(gt_mesh, num_samples=10_000, return_normals=True)
	pred_sample_dict = modified_sample(pred_mesh, num_samples=10_000, return_normals=True)

	# Calculate errors for reporting - by considering samples over the surface
	errs = modified_chamf(pred_sample_dict['verts'], gt_sample_dict['verts'],
						  x_normals=pred_sample_dict['normals'], y_normals=gt_sample_dict['normals'])

	# Calculate errors for visualisation - by considering every vertex
	vis_errs = modified_chamf(pred_mesh.verts_padded(), gt_mesh.verts_padded(),
								x_normals=pred_mesh.verts_normals_padded(), y_normals=gt_mesh.verts_normals_padded())

	# convert from cosine similarity to error in degrees
	errs['cham_norm_x'] = torch.rad2deg(torch.acos(errs['cham_norm_x']))
	errs['cham_norm_y'] = torch.rad2deg(torch.acos(errs['cham_norm_y']))
	vis_errs['cham_norm_x'] = torch.rad2deg(torch.acos(vis_errs['cham_norm_x'])) 
	vis_errs['cham_norm_y'] = torch.rad2deg(torch.acos(vis_errs['cham_norm_y']))

	if settings.get('model', 'FIND') == 'FIND':
		# apply masking here to not include errors for sole in pred -> real
		# errs has a sample of the vertices in, need to do correct indexing
		sampled_vertex_mask = FIND_sole_faces_mask[pred_sample_dict['face_idxs'].cpu().detach().numpy()[0]]
		errs['cham_x'][:, sampled_vertex_mask] = np.nan
		errs['cham_norm_x'][:, sampled_vertex_mask] = np.nan

		# vis_errs has all vertices in mesh in
		vis_errs['cham_x'][:, FIND_sole_vertex_mask] = np.nan
		vis_errs['cham_norm_x'][:, FIND_sole_vertex_mask] = np.nan


	# visualisation info for each metric of error
	vis_params = {
		'cham': dict(vmin=0, vmax=1e-4, mag=1_000_000, units='um', cutoffs=np.array([5, 10, 15, 20, 25])*1e-6, xscale='log'),
		'cham_norm': dict(vmin=0, vmax=60, mag=1, units='deg', cutoffs=[5, 7.5, 11.25, 22.5, 30], xscale='lin')
	}

	# define axes
	fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col')
	axs[0, 0].set_title('Chamfer Error')
	axs[0, 1].set_title('Normal Error')
	axs[0, 0].set_ylabel('pred2real')
	axs[1, 0].set_ylabel('real2pred')
	axs[1, 0].set_xlabel('um')
	axs[1, 1].set_xlabel('Degrees')
	axs[1,1].set_xlim(0, 90)
	axs[1, 1].set_yticks([0, 30, 60, 90])


	with Reporter(os.path.join(eval_dir, 'report.txt')) as report:
		report(f"Experiment: {exp_dir}")

		i = 0
		for L in ['cham', 'cham_norm']:
			report(L)
			table = PrettyTable()
			cutoffs = vis_params[L]['cutoffs']
			mag = vis_params[L]['mag']
			
			table.field_names = ['Desc', 'Mean', 'Median', 'RMSE'] + [f'% < {round(x*mag)}' for x in cutoffs]

			for desc, x in zip(['pred2real', 'real2pred'], ['x', 'y']):
				e = errs[f'{L}_{x}'].cpu().detach().numpy()
				e = e[~np.isnan(e)] # filter out nan values

				metrics = eval_metrics(e, cutoffs=cutoffs)

				table.add_row([desc] +
							   [f'{metrics[k] * mag:.2f}' for k in ['mean', 'median', 'rmse']] +
								[f'{i * 100:.1f}' for i in metrics['cutoffs']]
								)

				# plot distribution of errors
				ax = axs[i%2, i//2]
				if vis_params[L]['xscale'] == 'log':
					ax.hist(**get_loghist(np.ravel(e)*mag, 100), density=True)
					ax.set_xscale('log')

				else:
					ax.hist(np.ravel(e) * mag, bins=100, density=True)

				i+=1

				results[f'{L}_{desc}'] = {**{k: metrics[k] * mag for k in ['mean', 'median', 'rmse']}, 
								 **{f'% < {round(c*mag)}': i * 100 for c, i in zip(cutoffs, metrics['cutoffs'])}}

			report(table.get_string())
		report("")


	plt.savefig(os.path.join(eval_dir, 'err_dist.png'))
	plt.close()

	# Set up rendering
	if render:
		renderer: Renderer = Renderer(image_size=256, max_faces_per_bin=100_000, device=device)
		R, T = view_from(['side1', 'topdown', 'side2'])
		nviews = len(R)

		vis_elements = []

		# render chamf & norm err on GT mesh and pred mesh
		for i, (mesh, err_key) in enumerate(zip([gt_mesh, pred_mesh, gt_mesh, pred_mesh],
												['cham_y', 'cham_x', 'cham_norm_y', 'cham_norm_x'])):

			vis_type = 'cham_norm' if 'norm' in err_key else 'cham'

			# set texture according to error
			this_error = vis_errs[err_key]
			colours = err_to_colour(this_error, vmin=vis_params[vis_type]['vmin'], vmax=vis_params[vis_type]['vmax'])
			mesh.textures = TexturesVertex(colours)

			res = renderer(mesh, R, T, render_normals=False, render_sil=False) # render mesh

			# add to plot
			vis_elements.append([res['rgb'][n] for n in range(nviews)])

		grid = produce_grid(vis_elements)

		gridH, gridW, _ = grid.shape
		left_size = gridH // 8  # size of left padding in pix
		right_size = gridH // 8 # right size padding for colourbar
		out = np.zeros((gridH, left_size + gridW + right_size, 3), dtype=np.uint8)
		out[:, left_size:-right_size] = grid

		# write row names
		row_names = 'Chamf\nGT', 'Chamf\nFIND', 'Norm\nGT', 'Norm\nFIND'
		for i in range(4):
			out = put_text(out, row_names[i],
						x=0, y=int(gridH*i/4), width=int(left_size), height=int(gridH//4), scale=left_size / 100,
						vertical=True)

		# add colourbars
		# width, height, colours, points = (0, 1), orientation = 'vertical'
		cW, cH = right_size//2, int(gridH*0.3)
		cbar_x = left_size + gridW + (right_size - cW) // 2
		cbar_ys = [int(0.1 * gridH), int(0.6*gridH)]
		for key, y in zip(['cham', 'cham_norm'], cbar_ys):
			params = vis_params[key]
			cbar = colourbar(cW, cH, colours=((255, 0, 0), (0, 255, 0)))
			out[y:y + cH, cbar_x:cbar_x + cW] = cbar

			vmax_string = f"{params['vmax'] * params['mag']:g} {params['units']}"
			out = put_text(out, vmax_string, left_size + gridW, y, right_size, int(gridH*0.05), scale=gridH/1000)
			out = put_text(out, f"0 {params['units']}", left_size + gridW, y+cH, right_size, int(gridH * 0.05), scale=gridH/1000)

		cv2.imwrite(os.path.join(eval_dir, 'eval.png'), cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

	with open(os.path.join(exp_dir, 'results.json'), 'w') as outfile:
		json.dump(results, outfile)

	return results

def chunks(l, n):
	"""Yield n number of striped chunks from l."""
	return [l[i::n] for i in range(0, n)]

class EvalThread:
	def __init__(self, chunk, device='cuda:0', **kwargs):
		self.chunk = chunk
		self.device = device
		self.kwargs = kwargs
	
	def run(self):
		for f in self.chunk:
			eval_exp(f, **self.kwargs)

def eval_exps(exps_dir, render=True, gpus=0, set_context=True):
	"""Evaluate multiple experiments at once.
	Use threading if gpus given as list"""
	res = {}
	expts = sorted([os.path.join(exps_dir, f) for f in sorted(os.listdir(exps_dir)) if os.path.isdir(os.path.join(exps_dir, f))])
	
	if isinstance(gpus, int):
		EvalThread(expts, render=render).run()
		
	else:
		if set_context:
			mp.set_start_method('spawn')
		threads = []
		for gpu, chunk in zip(gpus, chunks(expts, len(gpus))):
			thread = EvalThread(chunk, device=f'cuda:{gpu}', render=render)
			threads.append(Process(target=thread.run))

		for thread in threads: thread.start()
		for thread in threads: thread.join() # hold here until finished

	# load all results once completed
	for ex in expts:
		res_loc = os.path.join(ex, 'results.json')
		if os.path.isfile(res_loc):
			with open(res_loc) as infile:
				res[os.path.split(ex)[-1]] = json.load(infile)


	# one table for each of mean, median, RMSE
	out = ''
	for metric in ['mean', 'median', 'rmse']:
		out+=f'{metric.upper()}\n'
		exps_table = PrettyTable() 
		exps_table.field_names = ['Exp', 'Cham p2r', 'Cham r2p', 'Norm p2r', 'Norm r2p']

		items = [f'{i}_{j}' for i in ['cham', 'cham_norm'] for j in ['pred2real', 'real2pred']]
		for exp_name, exp_res in res.items():
			exps_table.add_row([exp_name] + [f'{exp_res[i][metric]:.2f}' for i in items])

		exps_table.add_row(['AVG'] + [f'{np.mean([exp_res[i][metric] for exp_res in res.values()]):.2f}' for i in items])
		out += exps_table.get_string()
		out += '\n\n'


	with open(os.path.join(exps_dir, 'report.txt'), 'w') as outfile:
		outfile.write(out)
	
	with open(os.path.join(exps_dir, 'results.json'), 'w') as outfile:
		json.dump(res, outfile)