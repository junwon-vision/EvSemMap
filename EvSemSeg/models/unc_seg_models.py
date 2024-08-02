import torch
from torch import nn
import pdb
from .evidential_loss import EvidentialLossCal
from .metrics.mIoU_calculator import IoUs_calculator

from torchvision.models.segmentation import deeplabv3_resnet50


res50 = "" #TODO: Fix the resnet50 pretrained file path here.

class deeplabv3(nn.Module):
	def __init__(self, writer, n_classes, unc_args, void_index=None):
		super(deeplabv3, self).__init__()

		self.encoder = deeplabv3_resnet50(weights=None, num_classes=n_classes, weights_backbone=None)
		self.encoder.backbone.load_state_dict(torch.load(res50), strict=False)
		self.criterion = nn.CrossEntropyLoss(reduction='none')
		self.lossCal = EvidentialLossCal(writer=writer, unc_args=unc_args, void_index=void_index)

	def set_max_iter(self, max_iter):
		self.lossCal.max_iter = max_iter

	def forward(self, img, label, iter, epoch, with_acc_ece = False):
		x = self.encoder(img)['out']

		# Evidential Loss
		loss = self.lossCal.loss(x, label.long(), iter, epoch)

		if with_acc_ece:
			acc, ece = self.calculate_acc_ece(x, label)
			return loss, acc, ece
		return loss
	
	def normalize(self, target):
		target_norm = (target - target.min()) 
		return target_norm / target_norm.max()
	
	def calculate_acc_ece(self, logit, label):
		with torch.no_grad():
			alpha = self.logit_to_alpha(logit)
			pred  = torch.argmax(alpha, dim=1)
			num_classes = alpha.shape[1]
			S = torch.sum(alpha, dim=1, keepdim=False)
			certainty = 1 - num_classes / S # Basic Certainty

			pred, label, certainty = pred.cpu(), label.cpu(), certainty.cpu()

			isCorrect, certainty = (pred == label), certainty
			intersection, union = IoUs_calculator(label, pred, num_classes)
			# Certainty 2
			# max_alpha, _ = torch.max(alpha, dim=1)
			# certainty = max_alpha / S

			# S = S.unsqueeze(1)
			# diffE = torch.sum(torch.lgamma(alpha), dim=1, keepdim=False) - torch.lgamma(S) - torch.sum((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S)), dim=1, keepdim=False)
			# mutInfo = - torch.sum((alpha / S) *( torch.log( alpha / S ) - torch.digamma(alpha + 1) + torch.digamma(S + 1)), dim=1, keepdim=False)
			# aleaUnc = 1 - max_alpha / S
			
			acc = (pred == label).float().mean()
			# ece = torch.abs(certainty - (pred == label).float()).mean()
			
			# ece2 = torch.abs((1 - self.normalize(diffE)) - (pred == label).float()).mean()
			# ece3 = torch.abs((1 - self.normalize(mutInfo)) - (pred == label).float()).mean()
			# ece4 = torch.abs((1 - aleaUnc) - (pred == label).float()).mean()
			# ece5 = torch.abs((1 - self.normalize(aleaUnc)) - (pred == label).float()).mean()
			# ece6 = torch.abs(self.normalize(certainty) - (pred == label).float()).mean()
		return acc, intersection, union, isCorrect, certainty
		# return acc, ece, ece2, ece3, ece4, ece5, ece6
	
	def evaluate_uncertainty_measure(self, img, label):
		# mask_pred: ex. [3, 9, 2056, 2464]
        # labels: ex. [3, 1, 2056, 2464]		
		x = self.encoder(img)['out']
		return self.calculate_acc_ece(x, label)

	def logit_to_alpha(self, logit):
		return self.lossCal.logit_to_alpha(logit)
	
	def inference(self, img, uncMap=False) :

		x = self.encoder(img)['out']
		label = torch.argmax(x, dim=1)

		if uncMap:
			alpha = self.logit_to_alpha(x)
			# max_alpha, _ = torch.max(alpha, dim = 1)
			num_classes = x.shape[1]
			S = torch.sum(alpha, dim=1, keepdim=False)
			
			vacuity = num_classes / S
			
			# diffE = torch.sum(torch.lgamma(alpha), dim=1, keepdim=False) - torch.lgamma(S) - torch.sum((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S)), dim=1, keepdim=False)
			# mutInfo = - torch.sum((alpha / S) *( torch.log( alpha / S ) - torch.digamma(alpha + 1) + torch.digamma(S + 1)), dim=1, keepdim=False)
			# aleaUnc = 1 - max_alpha / S

			return label, vacuity
		
		return label
	
	def inference_logit(self, img) :

		x = self.encoder(img)['out']
	
		return x
	
	def inference_prob(self, img, visualization=False):
		# inference alpha
		if visualization:
			x = self.encoder(img)['out']
			alpha = self.logit_to_alpha(x)
			
			# label
			label = torch.argmax(alpha, dim=1)

			# uncertainty
			num_classes = x.shape[1]
			S = torch.sum(alpha, dim=1, keepdim=False)
			uncertainty = num_classes / S

			return alpha, label, uncertainty
		else:
			x = self.encoder(img)['out']
			return self.logit_to_alpha(x)
