"""Fit a model to data."""

import os
from collections import namedtuple, defaultdict
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import cv2
import imageio
from matplotlib import pyplot as plt

from data import FootScanDataset
from model import FIND
from utils import produce_grid, show_kps, show_kp_err, seg_overlap # visualisation tools
from utils.args import FitArgs, save_args
from utils import Renderer
from utils.forward import batch_to_device, calc_losses, LOSS_KEYS, KP_LOSSES, idx_batch
from utils.pytorch3d import export_mesh

Stage = namedtuple('Stage', 'name num_epochs lr params losses')
DEFAULT_STAGES = [
	Stage('Registration', 250, .001, ['reg'], ['kp_nll']),
	Stage('Deform verts', 250, .001, ['deform', 'reg'], ['norm_nll', 'sil', 'norm_nll']),
]



def visualize_view(batch, res, GT_mesh_data=None, norm_err=None):
	"""Visualize a reconstruction view given GT data, predicted data, and optionally GT_mesh data.
	Each must be in a dictionary of the data for just that view"""
	pred_kps_np = res['kps'].cpu().detach().numpy()

	gt_norm_vis =  batch['norm_rgb'] * batch['sil'].unsqueeze(-1) / 255 #  only consider norm RGB within silhouette for visualisation purposes

	vis_elements = [
				[batch['rgb'], batch['sil'], gt_norm_vis, show_kps(batch['rgb'], batch['kps'])],
				[res['rgb'], res['sil'], res['norm_rgb'], show_kps(res['rgb'], pred_kps_np, col=(0, 0, 255))],
				[None, seg_overlap(batch['sil'], res['sil']), norm_err, show_kp_err(batch['rgb'], batch['kps'], pred_kps_np)]
				]

	# if given a GT mesh, render this too.
	if GT_mesh_data is not None:
		gt_mesh_row = [GT_mesh_data['rgb'], GT_mesh_data['sil'], GT_mesh_data['norm_rgb'], None]
		vis_elements.insert(2, gt_mesh_row)

	return produce_grid(vis_elements)

def main(args):
	device = args.device
	exp_dir = os.path.join('exp', args.exp_name)
	os.makedirs(exp_dir, exist_ok=True)


	folder_names = dict(rgb=args.rgb_folder, norm=args.norm_folder, norm_unc=args.norm_unc_folder)
	dataset = FootScanDataset(args.data_folder, targ_img_size=args.targ_img_size, folder_names=folder_names)
	
	if args.restrict_num_views is not None:
		dataset.restrict_views(args.restrict_num_views)
	
	data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

	renderer = Renderer(image_size=args.targ_img_size, device=device, max_faces_per_bin=100_000,
		     cam_params=dataset.camera_params)

	model = FIND(args.find_pth,
				 kp_labels=dataset.kp_labels,
				 opt_posevec=not args.no_posevec_param).to(device)

	STAGES = getattr(args, 'stages', DEFAULT_STAGES)
	loss_weights = {k: getattr(args, f'weight_{k}') for k in LOSS_KEYS}
	losses_per_stage = []

	num_epochs = sum(s.num_epochs for s in STAGES)
	n_frames = min(250, num_epochs)
	vis_every = num_epochs // n_frames
	frames = []

	for stage_idx, stage in enumerate(STAGES):
		optimiser = torch.optim.Adam(model.get_params(stage.params), lr=stage.lr)



		if len([l for l in stage.losses if l in KP_LOSSES]) > 1:
			raise ValueError("Only one form of keypoint loss per stage is supported.")

		stage_loss_log = defaultdict(list)

		render_normals = 'norm_nll' in stage.losses or 'norm_al' in stage.losses
		render_sil = 'sil' in stage.losses or render_normals

		with tqdm(range(stage.num_epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tqdm_it:
			for epoch in tqdm_it:
				optimiser.zero_grad()

				is_frame_epoch = (epoch % vis_every) == 0 
				is_last_epoch =  epoch == stage.num_epochs - 1
				is_vis = is_frame_epoch or is_last_epoch

				epoch_losses = defaultdict(list)

				for nbatch, batch in enumerate(data_loader):
					batch = batch_to_device(batch, device)

					batch['sil'] = (batch['norm_alpha'] < args.alpha_threshold).float() # create silhouette from uncertainty map

					new_mesh = model()  # get Meshes prediction
					res = renderer(new_mesh, batch['R'], batch['T'], normals_fmt='ours',
								   render_rgb=is_vis,
								   render_normals=render_normals or is_vis,
								   render_sil=render_sil or is_vis,
								   keypoints=model.kps_from_mesh(new_mesh),
								   mask_out_faces=model.get_mask_out_faces())  # render the mesh

					res['new_mesh'] = new_mesh

					loss_dict = calc_losses(res, batch, stage.losses, {'img_size': args.targ_img_size})
					loss = sum(loss_dict[k] * loss_weights[k] for k in stage.losses)

					for k, v in loss_dict.items(): epoch_losses[k].append(v.item())
					epoch_avgs = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
					epoch_avg_loss = sum(epoch_avgs[k] * loss_weights[k] for k in stage.losses)

					tqdm_it.set_description(f'[{stage.name}] LOSS = {epoch_avg_loss:.4f} | ' + ', '.join(f'{k}: {v:.4f}' for k, v in epoch_avgs.items()))

					# Store for plotting graph of losses
					stage_loss_log['loss'].append(loss.item())
					for k, v in loss_dict.items(): stage_loss_log[k].append(v.item() * loss_weights[k])


					if is_last_epoch:
						for v in range(len(batch['R'])):
							view_img = visualize_view(batch=idx_batch(batch, v), res=idx_batch(res, v))
														# GT_mesh_data=None if gt_mesh_res is None else idx_batch(gt_mesh_res, v))
							
							if is_frame_epoch: # save to video
								pass

							if is_last_epoch: # save to image
								view_vis_dir = os.path.join(exp_dir, 'views', f'stage_{stage_idx:02d}')
								os.makedirs(view_vis_dir, exist_ok=True)
								overall_v = nbatch * args.batch_size + v
								cv2.imwrite(os.path.join(view_vis_dir, f'view_{overall_v:02d}.png'), cv2.cvtColor(view_img, cv2.COLOR_BGR2RGB))


		# STAGE END			

		export_mesh(model(), os.path.join(exp_dir, f'fit_{stage_idx:02d}.obj'), False)
		losses_per_stage.append(stage_loss_log)

	# if args.produce_video:
		# fps = 25
		# fname = os.path.join(exp_dir, f'vis.mp4')
		# imageio.mimwrite(fname, frames, fps=fps)
		# print(f"Video written to {fname}")

	# Plot graph of optimisation losses
	fig, axs = plt.subplots(nrows=len(STAGES))
	for n, stage_loss_info in enumerate(losses_per_stage):
		ax = axs[n]
		ax.set_title(STAGES[n].name)
		for k, v in stage_loss_info.items():
			ax.plot(v, label=k)
		ax.legend()

	plt.savefig(os.path.join(exp_dir, 'loss_plot.png'), dpi=200)
	plt.close()

	# save model params
	model.save(os.path.join(exp_dir, 'find_params.json'))

	save_args(args, os.path.join(exp_dir, 'opts.yaml'))

if __name__ == '__main__':
	parser = FitArgs()
	args = parser.parse()

	main(args)