import torch.nn as nn
import torch

import numpy as np

__all__ = ['SiameseNetwork']

class BranchNetwork(nn.Module):
    """
    Brach network
    """
    def __init__(self):
        """
        Input image size 180 x 320 (h x w)
        """
        super(BranchNetwork, self).__init__()
        layers = []
        in_channels = 1
        layers += [nn.Conv2d(in_channels, 4, kernel_size=7, stride=2, padding=3)]
        layers += [nn.LeakyReLU(0.1, inplace=True)]

        layers += [nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True)]

        layers += [nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True)]

        layers += [nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True)]

        layers += [nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(*[nn.Linear(6 * 10 * 16, 16)])

    def forward(self, x):
        x = self.network(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SiameseNetwork(nn.Module):
    """
    siamese network has left and right branches
    """
    def __init__(self, network):
        super(SiameseNetwork, self).__init__()
        self.network = network

    def _forward_one_branch(self, x):
        x = self.network(x)
        x = x.view(x.shape[0], -1)

        # L2 norm in dimension 1 (each row)
        x = nn.functional.normalize(x, p=2)
        return x

    def forward(self, x1, x2):
        x1 = self._forward_one_branch(x1)
        x2 = self._forward_one_branch(x2)
        return x1, x2

    def feature(self, x):
        return self._forward_one_branch(x)

    def feature_numpy(self, x):
        feat = self.feature(x)
        feat = feat.data
        feat = feat.cpu()
        feat = feat.numpy()
        
        if len(feat.shape) == 4:
            # N x C x 1 x 1
            
            feat = np.squeeze(feat, axis=(2, 3))
        else:
            # N x C
            assert len(feat.shape) == 2
        return feat


def ut():
    from contrastive_loss import ContrastiveLoss
    branch = BranchNetwork()
    siamese_network = SiameseNetwork(branch)

    criterion = ContrastiveLoss(margin=1.0)

    N = 2
    x1 = torch.randn(N, 1, 180, 320)
    x2 = torch.randn(N, 1, 180, 320)
    y1 = torch.randn(N, 1)
    y_zero = torch.zeros(N, 1)
    y_ones = torch.ones(N, 1)
    label = torch.where(y1 > 0, y_ones, y_zero)
    label = torch.squeeze(label)
    # print(label.shape)

    f1, f2 = siamese_network(x1, x2)
    print('f1 shape {}'.format(f1.shape))
    loss = criterion(f1, f2, label)
    print(loss)

if __name__ == '__main__':
    ut()






