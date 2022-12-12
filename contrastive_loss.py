import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)

        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + label
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        ) / 2.0
        return loss_contrastive


def threshold_contrastive_loss(output1: torch.Tensor, output2: torch.Tensor, m: float):
    """dist > m --> 1 else 0"""
    euclidean_distance = F.pairwise_distance(output1, output2)
    threshold = euclidean_distance.clone()
    threshold.data.fill_(m)
    return (euclidean_distance > threshold).float()

