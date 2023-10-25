import torch
import torch.nn as nn
import numpy as np

class NormalLoss(nn.Module):

	def calc_loss(self, render_norm: torch.FloatTensor, target_norm: torch.FloatTensor, target_kappa, mask: torch.FloatTensor=None):
		raise NotImplementedError
	
	def forward(self, render_norm: torch.FloatTensor, target_norm: torch.FloatTensor, target_kappa:torch.FloatTensor = None,
			 mask: torch.FloatTensor=None):
		"""Calculate normal loss.

		:param render_norm: [B x H x W x 3] tensor of rendered normals
		:param target_norm: [B x H x W x 3] tensor of target normals
		:param target_kappa: [B x H x W x 1] tensor of target kappa values
		:param mask: [B x H x W x 1] optional tensor of masks
		"""
		return self.calc_loss(render_norm, target_norm, target_kappa, mask)

class L1(NormalLoss):
	"""L1"""
	def calc_loss(self, render_norm, target_norm, mask=None, target_kappa=None):
		l1 = torch.sum(torch.abs(render_norm - target_norm), dim=3, keepdim=True)
		if mask is not None:
			l1 = l1[mask]
		return torch.mean(l1)


class L2(NormalLoss):
	"""L2"""
	def calc_loss(self, render_norm, target_norm, mask=None, target_kappa=None):
		l2 = torch.sum(torch.square(render_norm - target_norm), dim=3, keepdim=True)
		if mask is not None:
			l2 = l2[mask]
		return torch.mean(l2)


class AL(NormalLoss):
	"""Angular loss"""
	def calc_loss(self, render_norm, target_norm, mask=None, target_kappa=None):
		dot = torch.cosine_similarity(render_norm, target_norm, dim=3)
		if mask is not None:
			valid_mask = mask[:, :, :, 0].float() \
							* (dot.detach() < 0.999).float() \
							* (dot.detach() > -0.999).float()
			valid_mask = valid_mask > 0.0
			dot = dot[valid_mask]
		return torch.mean(torch.acos(dot))


class NLL(NormalLoss):
	"""Negative log likelihood loss"""
	def calc_loss(self, render_norm, target_norm, target_kappa, mask=None):
		dot = torch.cosine_similarity(render_norm, target_norm, dim=3)
		if mask is not None:
			valid_mask = mask[:, :, :, 0].float() \
							* (dot.detach() < 0.999).float() \
							* (dot.detach() > -0.999).float()
			valid_mask = valid_mask > 0.0
			dot = dot[valid_mask]
			target_kappa = target_kappa[valid_mask]
		else:
			target_kappa = target_kappa

		loss_pixelwise = - torch.log(torch.square(target_kappa) + 1) \
							+ target_kappa * torch.acos(dot) \
							+ torch.log(1 + torch.exp(-target_kappa * np.pi))
		return torch.mean(loss_pixelwise)

