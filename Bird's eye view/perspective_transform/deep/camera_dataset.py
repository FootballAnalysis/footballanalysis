import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import cv2 as cv
class CameraDataset(Dataset):
    def __init__(self,
                 pivot_data,
                 positive_data,
                 batch_size,
                 num_batch,
                 data_transform,
                 is_train=True):
        """
        :param pivot_data: N x 1 x H x W
        :param positive_data: N x 1 x H x W
        :param batch_size:
        :param num_batch:
        """
        super(CameraDataset, self).__init__()
        assert pivot_data.shape == positive_data.shape

        self.pivot_data = pivot_data
        self.positive_data = positive_data
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.data_transform = data_transform
        self.num_camera = pivot_data.shape[0]

        self.positive_index = []
        self.negative_index = []
        self.is_train = is_train

        if self.is_train:
            self._sample_once()

        if not self.is_train:
            # in testing, loop over all pivot cameras
            self.num_batch = self.num_camera//batch_size
            if self.num_camera%batch_size != 0:
                self.num_batch += 1

    def _sample_once(self):
        batch_size = self.batch_size
        num_batch = self.num_batch

        self.positive_index = []
        self.negative_index = []
        num = batch_size * num_batch
        c_set = set([i for i in range(self.num_camera)])
        for i in range(num):
            idx1, idx2 = random.sample(c_set, 2)  # select two indices in random
            self.positive_index.append(idx1)
            self.negative_index.append(idx2)

        assert len(self.positive_index) == num
        assert len(self.negative_index) == num

    def _get_test_item(self, index):
        """
        In testing, the label is hole-fill value, not used in practice.
        :param index:
        :return:
        """
        assert index < self.num_batch
        
        n,h, w = self.pivot_data.shape
        
        batch_size = self.batch_size

        start_index = batch_size * index
        end_index = min(start_index + batch_size, self.num_camera)
        bsize = end_index-start_index

        x = torch.zeros(1, 1, h, w)
        label_dummy = torch.zeros(bsize)

        for i in range(start_index, end_index):
           
            pivot = self.pivot_data[i]
            #pivot=Image.fromarray(pivot)            
            x[0,:,:,:] = self.data_transform(pivot)
            
            
            

        #x = torch.tensor(x, requires_grad=True)
        x = x.clone().detach().requires_grad_(True)
        
        return x, label_dummy

    def _get_train_item(self, index):
        """
        :param index:
        :return:
        """
        assert index < self.num_batch

        n, c, h, w = self.pivot_data.shape
        batch_size = self.batch_size

        start_index = batch_size * index
        end_index = start_index + batch_size
        positive_index = self.positive_index[start_index:end_index]
        negative_index = self.negative_index[start_index:end_index]

        x1 = torch.zeros(batch_size * 2, c, h, w)
        x2 = torch.zeros(batch_size * 2, c, h, w)
        label = torch.zeros(batch_size * 2)

        for i in range(batch_size):
            idx1, idx2 = positive_index[i], negative_index[i]
            pivot = self.pivot_data[idx1].squeeze()
            pos = self.positive_data[idx1].squeeze()
            neg = self.pivot_data[idx2].squeeze()

            pivot = Image.fromarray(pivot)
            pos = Image.fromarray(pos)
            neg = Image.fromarray(neg)

            # print('{} {} '.format(pivot.shape, self.data_transform(pivot).shape))
            x1[i * 2 + 0, :] = self.data_transform(pivot)
            x1[i * 2 + 1, :] = self.data_transform(pivot)
            x2[i * 2 + 0, :] = self.data_transform(pos)
            x2[i * 2 + 1, :] = self.data_transform(neg)

            label[i * 2 + 0] = 1
            label[i * 2 + 1] = 0
        #x1 = torch.tensor(x1, requires_grad=True)
        #x2 = torch.tensor(x2, requires_grad=True)
        x1 = x1.clone().detach().requires_grad_(True)
        x2 = x2.clone().detach().requires_grad_(True)
        return x1, x2, label


    def __getitem__(self, index):
        if self.is_train:
            return self._get_train_item(index)
        else:
            return self._get_test_item(index)

    def __len__(self):
        return self.num_batch

def ut():
    import scipy.io as sio
    import torchvision.transforms as transforms
    data = sio.loadmat('../../data/train_data_10k.mat')
    pivot_images = data['pivot_images']
    positive_images = data['positive_images']

    normalize = transforms.Normalize(mean=[0.0188],
                                     std=[0.128])

    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         normalize,
         ]
    )

    batch_size = 32
    num_batch = 64
    train_dataset = CameraDataset(pivot_images, positive_images, batch_size, num_batch, data_transform, is_train=True)

    for i in range(len(train_dataset)):
        x1, x2, label1 = train_dataset[i]
        print('{} {} {}'.format(x1.shape, x2.shape, label1.shape))
        break

    test_dataset = CameraDataset(pivot_images, positive_images, batch_size, num_batch, data_transform, is_train=False)

    for i in range(len(test_dataset)):
        x, _ = test_dataset[i]
        print('{}'.format(x.shape))
        break
    print('train, test dataset size {} {}'.format(len(train_dataset), len(test_dataset)))


if __name__ == '__main__':
    ut()

