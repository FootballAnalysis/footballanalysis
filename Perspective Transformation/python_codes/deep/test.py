import scipy.io as sio
import numpy as np

data = sio.loadmat('../../data/train_data_10k.mat')

print('{}'.format(data.keys()))
print('{}'.format(data['positive_images'].shape))
#cameras = data['cameras']
#features = data['features']
#print(data.keys())
#print('{} {}'.format(cameras.shape, features.shape))