"""Class to run experiments. Experiments will inherit from this, with a run() function
which takes some param and runs, supporting threading."""
import numpy as np
from multiprocessing import Process
import multiprocessing as mp
from .eval_utils import eval_exps
import os
from fit import main as fit_main
from fit import FitArgs


def chunks(l, n):
	"""Yield n number of striped chunks from l."""
	return [l[i::n] for i in range(0, n)]


class Thread:
	def __init__(self, exp: 'Experiment', chunk, device='cuda:0'):
		self.exp = exp
		self.chunk = chunk
		self.device = device

	def run(self):
		for f in self.chunk:
			self.exp.run(f, device=self.device)

class Experiment:
	def __init__(self, exp_dir=''):
		self.exp_dir = exp_dir

	def run(self, x, device='cuda:0'):
		raise NotImplementedError
	
	def threads(self, items:list, gpus:list):
		"""Split `items` across n gpus. Return list of threads"""
		threads = []
		for gpu, chunk in zip(gpus, chunks(items, len(gpus))):
			thread = Thread(self, chunk, device=f'cuda:{gpu}')
			threads.append(Process(target=thread.run))

		return threads
	
	def launch(self, items:list, gpus:list, eval=True):
		"""Launch experiment"""
		threads = self.threads(items, gpus)
		for thread in threads:
			thread.start()

		# Run evaluation once finished
		for thread in threads:
			thread.join() # hold here until thread finished

		if eval:
			eval_exps(os.path.join('exp', self.exp_dir), render=False, gpus=gpus,
				set_context=False) # evaluate, with threading (context already set)

class StandardExp(Experiment):
	def __init__(self, exp_dir='', data_folder:str='', scans: list = [], 
			  gpus: list = [0]):
		super().__init__(exp_dir)

		self.data_folder = data_folder
		self.launch(scans, gpus=gpus)

	def run(self, scan, device='cuda:0'):
		exp_name = scan

		parser = FitArgs()
		args = parser.parse(exp_name=f'{self.exp_dir}/{exp_name}',
					  data_folder = os.path.join(self.data_folder, scan),
					  include_gt_mesh=True,
					  restrict_num_views=10, # for faster runspeed
					  device=device)
		
		fit_main(args)

class RestrictumViewsExp(Experiment):
	def __init__(self, exp_dir='', data_folder:str='', scans: list = [], 
			  num_views: list = (3, 5, 10, 15), gpus: list = [0]):
		super().__init__(exp_dir)

		self.data_folder = data_folder

		items = [(scan, num_view) for scan in scans for num_view in num_views]
		self.launch(items, gpus=gpus)

	def run(self, item, device='cuda:0'):
		scan, num_views = item
		exp_name = f'{scan}_{num_views:02d}'

		parser = FitArgs()
		args = parser.parse(exp_name=f'{self.exp_dir}/{exp_name}',
					  data_folder = os.path.join(self.data_folder, scan),
					  restrict_num_views=num_views,
					  include_gt_mesh=True,
					  device=device)
		
		fit_main(args)


		