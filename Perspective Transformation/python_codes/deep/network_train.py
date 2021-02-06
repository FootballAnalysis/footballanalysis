import sys
#sys.path.append('../')

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import time
import random
import scipy.io as sio

import argparse

from contrastive_loss import ContrastiveLoss
from siamese import BranchNetwork, SiameseNetwork
from camera_dataset import CameraDataset
from util.synthetic_util import SyntheticUtil
"""
Train a siamese network by given image pairs and their labels
input: images and labels
output: network, feature
"""

parser = argparse.ArgumentParser()
parser.add_argument('--train-file', required=True, type=str, help='a .mat file')
parser.add_argument('--cuda-id', required=True, type=int, default=0, help='CUDA ID 0, 1, 2, 3')

parser.add_argument('--lr', required=True, type=float, default=0.01, help='learning rate')
parser.add_argument('--num-epoch', required=True, type=int, help='epoch number')
parser.add_argument('--batch-size', required=True, type=int)
parser.add_argument('--num-batch', required=True, type=int, help='training sample number')
parser.add_argument('--random-seed', required=True, type=int, help='random seed for generating train example')

parser.add_argument('--resume', default='', type=str, help='path to the save checkpoint')
parser.add_argument('--save-name', required=True, default='model.pth', type=str, help='model name .pth')

args = parser.parse_args()
train_file = args.train_file
cuda_id = args.cuda_id

learning_rate = args.lr
#step_size = args.step_size
num_epoch = args.num_epoch

batch_size = args.batch_size
num_batch = args.num_batch

random_seed = args.random_seed
resume = args.resume
save_name = args.save_name

print('random seed is {}'.format(random_seed))


normalize = transforms.Normalize(mean=[0.0188],
                                     std=[0.128])

data_transform = transforms.Compose(
    [transforms.ToTensor(),
     normalize,
     ]
)

# fix random seed
random.seed(random_seed)
try:
    data = sio.loadmat(r'C:\Users\mostafa\Desktop\test\SCCvSD-master\python\data_2/worldcup_sampled_cameras.mat')
except FileNotFoundError:
    print('Error: can not load .mat file from {}'.format(train_file))

######################################################################
pivot_cameras = data['pivot_cameras']
positive_cameras = data['positive_cameras']

data = sio.loadmat(r'C:\Users\mostafa\Desktop\test\SCCvSD-master\python\data_2/worldcup2014.mat')
print(data.keys())
model_points = data['points']
model_line_index = data['line_segment_index']

pivot_images, positive_images = SyntheticUtil.generate_database_images(pivot_cameras, positive_cameras,
                                                         model_points, model_line_index)


print("pivot and positive images loaded")
#############################################
#pivot_images = data['pivot_images']
#positive_images = data['positive_images']

n, c, h, w = pivot_images.shape
assert (h, w) == (180, 320)

print('Note: assume input image resolution is 180 x 320 (h x w)')

normalize = transforms.Normalize(mean=[0.0188],
                                 std=[0.128])

data_transform = transforms.Compose(
    [transforms.ToTensor(),
     normalize,
     ]
)

train_loader = CameraDataset(pivot_images,
                             positive_images,
                             batch_size,
                             num_batch,
                             data_transform,
                             is_train=True)
print('Randomly paired data are generated.')

# 2: load network
branch = BranchNetwork()
net = SiameseNetwork(branch)

criterion = ContrastiveLoss(margin=1.0)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                       lr=learning_rate,
                       weight_decay=0.000001)

# 3: setup computation device
if resume:
    if os.path.isfile(resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['state_dict'])
        print('resume from {}.'.format(resume))
    else:
        print('file not found at {}'.format(resume))
else:
    print('Learning from scratch')

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(cuda_id))
    net = net.to(device)
    criterion = ContrastiveLoss(margin=1.0).cuda(device)
    cudnn.benchmark = True

print('computation device: {}'.format(device))

def save_checkpoint(state, filename):
    file_path = os.path.join(filename)
    torch.save(state, file_path)

pdist = nn.PairwiseDistance(p=2)
for epoch in range(num_epoch):
    net.train()
    train_loader._sample_once()

    running_loss = 0.0
    running_num = 0
    start = time.time()

    positive_dist = 0.0
    negative_dist = 0.0

    for i in range(len(train_loader)):
        x1, x2, label = train_loader[i]

        x1, x2, label = x1.to(device), x2.to(device), label.to(device)
        feat1, feat2 = net(x1, x2)

        loss = criterion(feat1, feat2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_num += len(label)

        # distance
        dist = pdist(feat1, feat2)
        for j in range(len(label)):
            if label[j] == 1:
                positive_dist += dist[j]
            elif label[j] == 0:
                negative_dist += dist[j]
            else:
                assert 0

    assert running_num > 0
    stop = time.time()
    print('cost time: {:.1f}'.format(stop-start))

    running_loss = running_loss / running_num
    positive_dist = positive_dist / running_num
    negative_dist = negative_dist / running_num
    print('Epoch: {:d}, training loss {:.5f}'.format(epoch + 1, running_loss))
    print('Epoch: {:d}, positive distance {:.3f}, negative distance {:.3f}'.format(epoch + 1,
                                                                                   positive_dist,
                                                                                   negative_dist))
    dist_ratio = negative_dist / (positive_dist + 0.000001)
    print('Epoch: {:d}, training distance ratio {:.2f}'.format(epoch + 1, dist_ratio))

    # save model
    if (epoch + 1)%10 == 0:
        save_checkpoint({'epoch':epoch+1,
                         'state_dict':net.state_dict(),
                         'optimizer':optimizer.state_dict()},
                          save_name)
        print('save model to : {}'.format(save_name))

# move net to cpu
net = net.to('cpu')
save_checkpoint({'epoch':epoch+1,
                'state_dict':net.state_dict(),
                'optimizer':optimizer.state_dict()},
                save_name)
print('save model to : {}'.format(save_name))
print('Finished training')





