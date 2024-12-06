import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def one_hot(batch_size, label, class_num):
    label_cpu = label.cpu()
    view = label_cpu.size() + (1,)
    label_cpu = label_cpu.view(*view)

    y_one_hot = torch.zeros(batch_size, class_num).scatter_(1, label_cpu, 1)
    return y_one_hot


class FocalLoss(nn.Module):
    def __init__(self, weights, alpha=1, gamma=3, eps=1e-7, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.size_average = size_average
        self.weights = weights / weights.sum()

    def forward(self, input, target):
        batch_size = target.shape[0]
        y = one_hot(batch_size, target, input.size(-1))
        if torch.cuda.is_available():
            y = y.cuda()
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = -1 * y * torch.log(logit)  # cross entropy                     # -t*log(s1) - (1-t)*log(1-s1)

        loss = self.alpha * (1 - logit) ** self.gamma * loss  # focal loss
        loss = loss * self.weights
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
