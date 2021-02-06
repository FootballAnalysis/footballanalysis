
# contrastive loss from "Dimensionality Reduction by Learning an Invariant Mapping"
# modified from https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7

import torch
import torch.nn as nn

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin):
        """
        :param margin: 1.0
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label):
        """
        :param x1: N * D
        :param x2: N * D
        :param label: 0 for un-similar, 1 for similar
        :return:
        """
        assert len(x1.shape) == 2
        assert len(x2.shape) == 2
        assert len(label.shape) == 1
        assert label.shape[0] == x1.shape[0]

        pdist = nn.PairwiseDistance(p=2)
        dist = pdist(x1, x2)
        loss = label * torch.pow(dist, 2) \
               + (1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        loss = torch.sum(loss)

        return 0.5 * loss

def ut_contrastive_loss():
    closs = ContrastiveLoss(margin=1.0)
    N = 10
    x1 = torch.randn(N, 256)
    x2 = torch.randn(N, 256)
    y1 = torch.randn(N, 1)
    y_zero = torch.zeros(N, 1)
    y_ones = torch.ones(N, 1)
    label = torch.where(y1>0, y_ones, y_zero)
    label = torch.squeeze(label)
    #print(label.shape)

    loss = closs(x1, x2, label)
    print(loss.shape)
    print(loss)

if __name__ == '__main__':
    ut_contrastive_loss()



