import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pdb

def get_scale_label(y_true, params = None):
    if params is not None:
        y_true = y_true.float() / 255.0 * params['scale']
    return y_true

def binary_crossentropy(y_true, y_pred, annotate_mask=None, weight=1.0):
    if annotate_mask is not None:
        sumit = torch.zeros_like(y_pred)
        nN, nChs, nD, nH, nW = y_pred.size()
        for i in range(nN):
            loss = -1.0 * weight*y_true[i]*torch.log(torch.sigmoid(y_pred[i]) +1e-10) - (1-y_true[i]) * torch.log(1-torch.sigmoid(y_pred[i])+1e-10)
            if hasattr(y_pred, 'requires_grad'):
                if torch.sum(annotate_mask[i]).data.cpu().numpy() > 0:
                    sumit[i] = loss * annotate_mask[i] * (nChs * nD * nH * nW / torch.sum(annotate_mask[i]))
                else:
                    sumit[i] = loss * annotate_mask[i]
            else:
                if torch.sum(annotate_mask[i]) > 0:
                    sumit[i] = loss * annotate_mask[i] * (nChs * nD * nH * nW / torch.sum(annotate_mask[i]))
                else:
                    sumit[i] = loss * annotate_mask[i]
    else:
        sumit = -1.0 * weight*y_true*torch.log(torch.sigmoid(y_pred) + 1e-10) - (1-y_true) * torch.log(1-torch.sigmoid(y_pred) + 1e-10)
    return torch.mean(sumit)

def weighted_loss3D(y_true, y_pred, annotate_mask=None, weight=1.0):
    assert y_pred.dim() == 5, 'dimension is not matched!!'
    if y_true.dim() == 4:
        y_true = y_true.unsqueeze(1)
    if annotate_mask is not None:
        if annotate_mask.dim() == 4:
            annotate_mask = annotate_mask.unsqueeze(1)
        annotate_mask = annotate_mask.float() / 255.0 # normalize values to [0, 1]
    masked_loss = binary_crossentropy(y_true, y_pred, annotate_mask=annotate_mask, weight=weight)
    return masked_loss

def wbce_loss(y_true, y_pred, annotate_mask=None, weight=1.0, model3D=True):
    return weighted_loss3D(y_true, y_pred, annotate_mask=annotate_mask, weight=weight)
