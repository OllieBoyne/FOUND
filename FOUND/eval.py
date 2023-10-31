"""Script for running a large number of fits and evaluating the results
against ground-truth data, as used in results for the FOUND paper.

Uses multi-threading and multi-gpu"""

import multiprocessing as mp
import os
from utils.experiment import StandardExp
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_folder', type=str, default='data/scans')
parser.add_argument('--exp_name', type=str, default='exp_on_all')
parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parser.parse_args()

    main_data_folder = args.data_folder
    scans = os.listdir(main_data_folder)

    StandardExp(exp_dir=args.exp_name, 
                data_folder=main_data_folder,
                scans=scans, gpus=args.gpus)