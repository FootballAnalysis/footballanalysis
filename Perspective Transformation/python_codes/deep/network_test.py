import sys
sys.path.append('../')
import os

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import time
import numpy as np
import scipy.io as sio

import argparse


from siamese import BranchNetwork, SiameseNetwork
from camera_dataset import CameraDataset
from util.synthetic_util import SyntheticUtil
"""
Extract feature from a siamese network
input: network and edge images
output: feature and camera
"""

parser = argparse.ArgumentParser()
parser.add_argument('--edge-image-file', required=True, type=str, help='a .mat file')
parser.add_argument('--model-name', required=True, type=str, help='model name .pth')
parser.add_argument('--batch-size', required=True, type=int)
parser.add_argument('--cuda-id', required=True, type=int, default=0, help='CUDA ID 0, 1, 2, 3')
parser.add_argument('--save-file', required=True, type=str, help='.mat file with')

args = parser.parse_args()
edge_image_file = args.edge_image_file


batch_size = args.batch_size
model_name = args.model_name
cuda_id = args.cuda_id
save_file = args.save_file

normalize = transforms.Normalize(mean=[0.0188],
                                     std=[0.128])

data_transform = transforms.Compose(
    [transforms.ToTensor(),
     normalize,
     ]
)


# 1: load edge image
try:
    data = sio.loadmat(r'C:\Users\mostafa\Desktop\test\SCCvSD-master\python\data_2/worldcup_sampled_cameras.mat')

except FileNotFoundError:
    print('Error: can not load .mat file from {}'.format(edge_image_file))

pivot_cameras = data['pivot_cameras']
positive_cameras = data['positive_cameras']
cameras = pivot_cameras
data = sio.loadmat(r'C:\Users\mostafa\Desktop\test\SCCvSD-master\python\data_2/worldcup2014.mat')
print(data.keys())
model_points = data['points']
model_line_index = data['line_segment_index']

pivot_images, positive_images = SyntheticUtil.generate_database_images(pivot_cameras, positive_cameras,
                                                         model_points, model_line_index)



#pivot_images = data['pivot_images']
#positive_images = data['positive_images']  # not actually used


n, c, h, w = pivot_images.shape
assert (h, w) == (180, 320)

print('Note: assume input image resolution is 180 x 320 (h x w)')
data_loader = CameraDataset(pivot_images,
                            positive_images,
                            batch_size,
                            -1,
                            data_transform,
                            is_train=False)
print('load {} batch edge images'.format(len(data_loader)))

# 2: load network
branch = BranchNetwork()
net = SiameseNetwork(branch)

if os.path.isfile(model_name):
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['state_dict'])
    print('load model file from {}.'.format(model_name))
else:
    print('Error: file not found at {}'.format(model_name))
    sys.exit()

# 3: setup computation device
device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(cuda_id))
    net = net.to(device)
    cudnn.benchmark = True
print('computation device: {}'.format(device))

features = []
with torch.no_grad():
    for i in range(len(data_loader)):
        x, _ = data_loader[i]
        x = x.to(device)
        feat = net.feature_numpy(x) # N x C

        features.append(feat)
        # append to the feature list

        if i%100 == 0:
            print('finished {} in {}'.format(i+1, len(data_loader)))

features = np.vstack((features))
print('feature dimension {}'.format(features.shape))

sio.savemat(save_file, {'features':features,
                        'cameras':cameras})
print('save to {}'.format(save_file))
