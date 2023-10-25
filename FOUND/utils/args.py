import argparse
import json
import os
import yaml

class FitArgs(argparse.ArgumentParser):
	def __init__(self):
		super().__init__()

		self.add_argument('--cfg', type=str, default=None, help="Path to .yaml file (overrides all other args)")

		self.add_argument('--device', type=str, default='cuda')
		self.add_argument('--exp_name', default='foot2foot', type=str, help="Experiment name")

		# DATA PARAMS
		self.add_argument('--data_folder', type=str, default='test_imgs/foot')
		self.add_argument('--targ_img_size', default=None, type=int,
							help='Resize images as close to this size(preserving aspect ratio) as possible.')
		self.add_argument('--include_gt_mesh', action='store_true', help="Use GT mesh while validating."
																		"Note: must be in data_folder, as mesh.obj")

		# Folder names - change these if your folder structure is different to the default
		self.add_argument('--rgb_folder', type=str, default='rgb', help="Name of folder containing RGB images")
		self.add_argument('--norm_folder', type=str, default='normal', help="Name of folder containing normal predictions")
		self.add_argument('--norm_unc_folder', type=str, default='norm_unc')

		# MODEL PARAMS
		self.add_argument('--model', type=str, default='FIND', choices=['scan', 'FIND'])
		self.add_argument('--keypoints_loc', type=str, default='misc/keypoints.csv')

		# scan opts
		self.add_argument('--scan_loc', default='misc/0015-A.obj')

		# FIND opts
		self.add_argument('--find_pth', type=str, default='data/find_nfap', help="Path to FIND model directory")
		self.add_argument('--no_posevec_param', action='store_true', help="Do not optimise over pose")

		# VIS PARAMS
		self.add_argument('--max_views_display', default=3, type=int, help="Maximum views to show in visualizer")


		# OPTIM OPTIONS
		self.add_argument('--views_per_iter', default=10, type=int, help="Sample a smaller number of views per iter to speed up runtime. Set this to None to turn off this feature.")
		self.add_argument('--restrict_num_views', default=None, type=int, help="Restrict to a fixed number of views. Set this to None to turn off this feature.")

		# OPTIM_WEIGHTS
		_defaults = dict(sil=1., norm=0.1, smooth=2e3,
		   	kp=0.5, kp_l1=0.5, kp_l2=0.5, kp_nll=0.5,
		    edge=1., norm_nll=0.1, norm_al=0.1)
		for k, v in _defaults.items():
			self.add_argument(f'--weight_{k}', default=v, type=float, help=f"Weight for `{k}` loss")


	def parse(self, **kwargs):
		"""Parse command line arguments, overwriting with anything from kwargs"""
		args = super().parse_args()
		for k, v in kwargs.items():
			setattr(args, k, v)

		if args.cfg is not None:

			with open(args.cfg, 'r') as f:
				cfg = yaml.safe_load(f)

			for k, v in cfg.items():
				setattr(args, k, v)

		return args


def is_jsonable(x):
	try:
		json.dumps(x)
		return True
	except (TypeError, OverflowError):
		return False

def save_args(args: argparse.Namespace, loc: str):
	# save arguments
	args_dict = {}
	for k, v in vars(args).items():
		if is_jsonable(v):
			args_dict[k] = v

	with open(loc, 'w') as outfile:
		json.dump(args_dict, outfile)