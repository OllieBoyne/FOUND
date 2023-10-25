import torch
from . import normal_losses
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss


KP_LOSSES = ['kp_l1', 'kp_l2', 'kp_nll']
NORM_LOSSES = ['norm_al', 'norm_nll']
LOSS_KEYS = ['sil', 'smooth', 'edge'] + NORM_LOSSES + KP_LOSSES

mse_loss = torch.nn.MSELoss(reduction='none')
normal_crit_al = normal_losses.AL()
normal_crit_nll = normal_losses.NLL()


def idx_batch(batch: dict, idx: int):
	"""Return a view of batch, sampling all tensors at idx"""
	return {k: v[idx] for k, v in batch.items() if torch.is_tensor(v)}

def batch_to_device(batch: dict, device: str):
	"""Return a copy of batch, with all tensors moved to device"""
	return {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

def calc_losses(res: dict, batch: dict, loss_list: list, aux: dict) -> dict:
	"""Run forward pass of all losses, return dict of loss values.
	
	:param res: dict of outputs from model
	:param batch: dict of inputs to model
	:param loss_list: list of losses to calculate
	:param aux: dict of auxiliary data
	
	"""
	loss_dict = {}

	if 'sil' in loss_list:
		loss_dict['sil'] = mse_loss(res['sil'], batch['sil']).mean()

	if 'norm_al' in loss_list:
		norm_mask = torch.logical_and(res['sil'] > 0, batch['sil'] > 0).unsqueeze(-1)
		loss_dict['norm_al'] = normal_crit_al(res['norm_xyz'], batch['norm_xyz'], mask = norm_mask)

	if 'norm_nll' in loss_list:
		norm_mask = (res['sil'] > 0).unsqueeze(-1)
		loss_dict['norm_nll'] = normal_crit_nll(res['norm_xyz'],
												batch['norm_xyz'],
												target_kappa = batch['norm_kappa'],
												mask = norm_mask)

	if 'smooth' in loss_list:
		loss_dict['smooth'] = mesh_laplacian_smoothing(res['new_mesh'], method="uniform")

	if 'edge' in loss_list:
		loss_dict['edge'] = mesh_edge_loss(res['new_mesh'])

	if any(k in loss_list for k in KP_LOSSES):
		y = res['kps']
		mu, visibility = batch['kps'][..., :2], batch['kps'][..., 2:]
		var = batch['kps_unc']

		if 'kp_l1' in loss_list:
			loss_kp = torch.abs(y - mu)
			key = 'kp_l1'

		elif 'kp_l2' in loss_list:
			loss_kp = torch.square(y - mu)
			key = 'kp_l2'

		elif 'kp_nll' in loss_list:
			loss_kp = (torch.square(y - mu) / (var + 1e-10))
			key = 'kp_nll'

		else:
			raise ValueError("Invalid keypoint loss")

		H, W = aux['img_size']
		loss_dict[key] = (loss_kp * visibility).mean() / ((H + W) / 2)

	return loss_dict