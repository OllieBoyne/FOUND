import argparse
import json
import os
import yaml


class FitArgs(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        self.add_argument(
            "--cfg",
            type=str,
            default=None,
            help="Path to .yaml file (overrides all other args)",
        )

        self.add_argument("--device", type=str, default="cuda")
        self.add_argument(
            "--exp_name", default="unnamed", type=str, help="Experiment name"
        )

        self.add_argument(
            "--alpha_threshold",
            default=30.0,
            type=float,
            help="Alpha threshold for silhouette (degrees)",
        )

        # DATA PARAMS
        self.add_argument("--data_folder", type=str, default="test_imgs/foot")
        self.add_argument(
            "--targ_img_size",
            default=(192, 144),
            type=tuple,
            help="(H, W) - Resize images to this size.",
        )
        self.add_argument(
            "--include_gt_mesh",
            action="store_true",
            help="Use GT mesh while validating."
            "Note: must be in data_folder, as mesh.obj",
        )

        # Folder names - change these if your folder structure is different to the default
        self.add_argument(
            "--rgb_folder",
            type=str,
            default="rgb",
            help="Name of folder containing RGB images",
        )
        self.add_argument(
            "--norm_folder",
            type=str,
            default="norm",
            help="Name of folder containing normal predictions",
        )
        self.add_argument("--norm_unc_folder", type=str, default="norm_unc")

        # FIND opts
        self.add_argument(
            "--find_pth",
            type=str,
            default="data/find_nfap",
            help="Path to FIND model directory",
        )
        self.add_argument(
            "--no_posevec_param", action="store_true", help="Do not optimise over pose"
        )

        # VIS PARAMS
        self.add_argument(
            "--max_views_display",
            default=3,
            type=int,
            help="Maximum views to show in visualizer",
        )
        self.add_argument(
            "--produce_video",
            action="store_true",
            help="Produce video of visualizations (slows performance)",
        )

        # OPTIM OPTIONS
        self.add_argument("--batch_size", default=12, type=int)
        self.add_argument(
            "--restrict_num_views",
            default=None,
            type=int,
            help="Restrict to a fixed number of views. Set this to None to turn off this feature.",
        )

        # OPTIM_WEIGHTS
        _defaults = dict(
            sil=1.0,
            norm=0.1,
            smooth=2e3,
            kp=0.5,
            kp_l1=0.5,
            kp_l2=0.5,
            kp_nll=0.5,
            edge=1.0,
            norm_nll=0.1,
            norm_al=0.1,
        )
        for k, v in _defaults.items():
            self.add_argument(
                f"--weight_{k}", default=v, type=float, help=f"Weight for `{k}` loss"
            )

        # MISC
        self.add_argument(
            "--fast", action="store_true", help="Fast mode (for evaluation only)"
        )

    def parse(self, **kwargs):
        """Parse command line arguments, overwriting with anything from kwargs"""
        args = super().parse_args()
        args.stages = None

        for k, v in kwargs.items():
            if not hasattr(args, k):
                raise ValueError(f"Unknown argument {k}")

            setattr(args, k, v)

        if args.cfg is not None:
            with open(args.cfg, "r") as f:
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

    with open(loc, "w") as outfile:
        json.dump(args_dict, outfile, indent=4)
