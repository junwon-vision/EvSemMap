import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import ceil
import pdb

class EvidentialLossCal():
    def __init__(self, writer, unc_args, void_index=None, max_epoch=60):
        ## Evidential Loss Params
        self.evd_type = unc_args['evd_type']
        self.unc_act = unc_args['unc_act']
        self.unc_type = unc_args['unc_type']
        self.kl_strength = unc_args['kl_strength']
        self.ohem = unc_args['ohem']

        # setting
        if self.unc_act == 'exp':
            self.activation = lambda x: torch.exp(torch.clamp(x, -10, 10)) # default, 10.
        elif self.unc_act == 'relu':
            self.activation = torch.relu
        elif self.unc_act == 'softplus':
            self.activation = F.softplus
        else:
            raise NotImplementedError

        if self.unc_type == 'digamma':
            self.unc_fn = torch.digamma
        elif self.unc_type == 'log':
            self.unc_fn = torch.log
        else:
            raise NotImplementedError
        ##########################################################

        self.epoch = 0
        self.iter = 0
        self.total_iter = 0
        self.max_iter = 1 # set by deeplabv3.set_max_iter
        self.max_epoch = max_epoch
        self.ignore_index = void_index

        if self.ohem is not None:
            assert (self.ohem >= 0 and self.ohem < 1)
        self.writer = writer

    def logit_to_evidence(self, logit):
        return self.activation(logit)

    def evidence_to_alpha(self, evidence):
        return evidence + 1.0

    def logit_to_alpha(self, logit):
        return self.evidence_to_alpha( self.logit_to_evidence(logit) )
    
    def expand_onehot_labels(self, label, target):
        assert label.dim() == 4 # label : [B, 1, W, H]
        # expanded = torch.zeros_like(target).scatter_(1, label, 1)
        # return expanded[:, :, :, :] 

        bin_labels = label.new_zeros(target.shape)
        valid_mask = (label >= 0) & (label != self.ignore_index)
        inds = torch.nonzero(valid_mask, as_tuple=True)
        if inds[0].numel() > 0:
            bin_labels[inds[0], label[valid_mask], inds[2], inds[3]] = 1

        return bin_labels
    
    def loss(self, mask_pred, labels, curr_iter, curr_epoch):
        # 1. Check Validity ###########################################################################################################
        # mask_pred  : [B, C, H, W] (ex. [3, 9, 2056, 2464])
        # labels     : [B, 1, H, W] or [B, H, W] (ex. [3, 1, 2056, 2464] or [3, 2056, 2464])
        if len(labels.shape) == 3:
            labels = labels.unsqueeze(1)
                    
        assert len(mask_pred.shape) == 4 and len(labels.shape) == 4
        assert labels.shape[1] == 1 and mask_pred.shape[0] == labels.shape[0] and mask_pred.shape[2] == labels.shape[2] and mask_pred.shape[3] == labels.shape[3]        
        # labels     : [B, 1, H, W]
        # curr_iter  : int
        # curr_epoch : int
        
        self.iter, self.epoch = curr_iter, curr_epoch
        self.total_iter += 1

        labels = labels.long() # labels: [3, 1, 2056, 2464]
        ###############################################################################################################################

        # 2. Prepare Values ###########################################################################################################
        labels_1hot = self.expand_onehot_labels(labels, mask_pred) # labels_1hot: [3, 9, 2056, 2464]
        alpha = self.logit_to_alpha(mask_pred) # mask_pred: [32, 9, 550, 688]
        alpha0 = torch.sum(alpha, dim=1, keepdim=True) # alpha0: [32, 1, 550, 688]
        
        edl_loss, loss_kl = 0.0, 0.0
        ###############################################################################################################################

        # 3(1) Main Loss. Evidential CE loss(digamma) or Evidential Log loss(log)
        edl_loss = torch.sum(labels_1hot * (self.unc_fn(alpha0) - self.unc_fn(alpha)), dim=1, keepdim=True)
        
        if self.ohem is not None:
            top_k = int(ceil(edl_loss.numel() * self.ohem))
            if top_k != edl_loss.numel():
                edl_loss, _ = edl_loss.topk(top_k)
        
        self.writer.add_scalar("Loss/evid_loss", edl_loss.view(-1).mean(), self.total_iter)
        ###############################################################################################################################
            
        # 3(2) KL Regularizer Loss.
        target_c = 1.0 # Hyper-Parameter!        
        
        kl_alpha = (alpha - target_c) * (1 - labels_1hot) + target_c
        kl_coef = self.kl_strength * (curr_epoch / self.max_epoch)

        loss_kl = self.compute_kl_loss(kl_alpha)
        self.writer.add_scalar("Loss/incorrect_reg_loss", loss_kl, self.total_iter)
        ###############################################################################################################################
        loss_semantic_seg = edl_loss + kl_coef * loss_kl
        self.writer.add_scalar("Loss/entire_loss", loss_semantic_seg.view(-1).mean(), self.total_iter)

        self.writer.flush()
        return loss_semantic_seg

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels
    
    def mean(self, l, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    #### IEDL related functions #####################################################################################################################

    def dirichlet_kl_divergence(self, alphas, target_alphas):
        epsilon = torch.tensor(1e-8)

        alp0 = torch.sum(alphas, dim=1, keepdim=True)
        target_alp0 = torch.sum(target_alphas, dim=1, keepdim=True)

        alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
        alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
        assert torch.all(torch.isfinite(alp0_term)).item()

        alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                                + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                            torch.digamma(alp0 + epsilon)), dim=1, keepdim=True)
        alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
        assert torch.all(torch.isfinite(alphas_term)).item()

        loss = torch.squeeze(alp0_term + alphas_term).mean()

        return loss
    

    def compute_kl_loss(self, alphas, labels=None, target_concentration=1.0, concentration=1.0, reverse=True):
        # Create array of target (desired) concentration parameters

        target_alphas = torch.ones_like(alphas) * concentration
        if labels is not None:
            target_alphas += torch.zeros_like(alphas).scatter_(1, labels, target_concentration - 1)

        if reverse:
            loss = self.dirichlet_kl_divergence(alphas, target_alphas)
        else:
            loss = self.dirichlet_kl_divergence(target_alphas, alphas)

        return loss
