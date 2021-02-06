import sys
#sys.path.append('../')
import os

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import time
import numpy as np
import scipy.io as sio
import cv2
import argparse


from siamese import BranchNetwork, SiameseNetwork
from camera_dataset import CameraDataset

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
    
    #data = sio.loadmat(r'C:\Users\mostafa\Desktop\Football_video_analyses\camera calibration\6\SCCvSD-master\SCCvSD-master\data\features\testset_feature.mat')
    #pivot_images=data['edge_map']
    #pivot_image=pivot_images[:,:,0,0]
    pivot_image = cv2.imread(edge_image_file,0)
    
except FileNotFoundError:
    print('Error: can not load .mat file from {}'.format(edge_image_file))


cv2.imshow("ss",pivot_image)
cv2.waitKey()
print(pivot_image.shape)
pivot_image=cv2.resize(pivot_image,(320,180),interpolation=1)
cv2.imshow("ss",pivot_image)
cv2.waitKey()

pivot_image=np.reshape(pivot_image,(1,1,pivot_image.shape[0],pivot_image.shape[1]))
print(pivot_image.shape)
print('Note: assume input image resolution is 180 x 320 (h x w)')

data_loader = CameraDataset(pivot_image,
                            pivot_image,
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
        print(x.shape)
        feat = net.feature_numpy(x) # N x C

        features.append(feat)
        # append to the feature list

        if i%100 == 0:
            print('finished {} in {}'.format(i+1, len(data_loader)))

    

features = np.vstack((features))
print('feature dimension {}'.format(features.shape))

sio.savemat(save_file, {'features':features,
                        'edge_map':pivot_image})
print('save to {}'.format(save_file))
