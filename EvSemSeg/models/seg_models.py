import torch
from torch import nn
import torch.nn.functional as F
import pdb
from torchvision.models.segmentation import deeplabv3_resnet50

from .metrics.mIoU_calculator import IoUs_calculator

res50 = "" #TODO: Fix the resnet50 pretrained file path here.
class deeplabv3(nn.Module):
    def __init__(self, writer, n_classes, void_index = None):
        super(deeplabv3, self).__init__()


        self.encoder = deeplabv3_resnet50(weights=None, num_classes=n_classes, weights_backbone=None)
        self.encoder.backbone.load_state_dict(torch.load(res50), strict=False)
        if void_index is not None:
            self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=void_index)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.writer = writer
        self.total_iter = 1


    def forward(self, img, label, with_acc_ece = False):
        
        x = self.encoder(img)['out']

        if len(label.shape) == 4:
            label = label.squeeze(1)
        
        loss = self.criterion(x, label.long())

        self.writer.add_scalar("Loss/vanilla_deeplabv3_loss", loss.mean(), self.total_iter)
        self.writer.flush()
        self.total_iter += 1

        if with_acc_ece:
            acc, ece = self.calculate_acc_ece(x, label)
            return loss, acc, ece
        return loss
    
    def calculate_acc_ece(self, logit, label):
        with torch.no_grad():
            prob = torch.nn.functional.softmax(logit, dim=1)
            pred = torch.argmax(prob, dim=1)
            num_classes = prob.shape[1]
            
            max_prob, _ = torch.max(prob, dim=1)
            certainty = max_prob

            acc = (pred == label).float().mean()

            pred, label, certainty = pred.cpu(), label.cpu(), certainty.cpu()
            isCorrect, certainty = (pred == label), certainty
            intersection, union = IoUs_calculator(label, pred, num_classes=num_classes)
            # expected_calibaration_error = torch.abs(certainty - (pred == label).float()).mean()

        return acc, intersection, union, isCorrect, certainty

    def evaluate_uncertainty_measure(self, img, label):
        # mask_pred: ex. [3, 9, 2056, 2464]
        # labels: ex. [3, 1, 2056, 2464]		
        x = self.encoder(img)['out']

        return self.calculate_acc_ece(x, label)
        
    
    def inference(self, img, uncMap=False) :

        x = self.encoder(img)['out']
        prob = torch.nn.functional.softmax(x, dim=1)
        label = torch.argmax(x, dim=1)

        if uncMap:
            max_prob, _ = torch.max(prob, dim=1)
            uncertainty = 1 - max_prob
            # entropy = torch.sum(-prob * torch.log2(prob), dim=1)
            return label, uncertainty
                
        return label

    def inference_logit(self, img) :

        x = self.encoder(img)['out']
    
        return x
    
    def inference_prob(self, img, visualization=False):
        if visualization:
            x = self.encoder(img)['out']
            prob = torch.nn.functional.softmax(x, dim=1)
            label = torch.argmax(x, dim=1)

            max_prob, _ = torch.max(prob, dim=1)
            uncertainty = 1 - max_prob
            # entropy = torch.sum(-prob * torch.log2(prob), dim=1)
            return prob, label, uncertainty
        else:
            x = self.encoder(img)['out']
            prob = torch.nn.functional.softmax(x, dim=1)
            return prob