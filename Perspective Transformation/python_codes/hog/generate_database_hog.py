# HoG feature for nearest neighbor search
import sys
sys.path.append('../')

import cv2 as cv
import scipy.io as sio
import numpy as np
from util.synthetic_util import SyntheticUtil


# HoG parameters
win_size = (128, 128)
block_size = (32, 32)
block_stride = (32, 32)
cell_size = (32, 32)
n_bins = 9


im_h, im_w = 180, 320
save_file = 'database_camera_feature_HoG.mat'

hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)

# database camera
data = sio.loadmat('../../data/worldcup_sampled_cameras.mat')
database_cameras = data['pivot_cameras']

n, _ = database_cameras.shape

# World Cup soccer template
data = sio.loadmat('../../data/worldcup2014.mat')
model_points = data['points']
model_line_index = data['line_segment_index']

database_features = []
for i in range(n):
    edge_image = SyntheticUtil.camera_to_edge_image(database_cameras[i,:],
                                                    model_points,
                                                    model_line_index,
                                                    im_h=720, im_w=1280,
                                                    line_width=4)
    edge_image = cv.resize(edge_image, (im_w, im_h))
    edge_image = cv.cvtColor(edge_image, cv.COLOR_BGR2GRAY)
    feat = hog.compute(edge_image)
    database_features.append(feat)
    if i%1000 == 0:
        print('finished {} examples in {}'.format(i+1, n))

database_features = np.squeeze(np.asarray(database_features), axis=2)

print('feature dimension {}'.format(database_features.shape))

sio.savemat(save_file, {'features':database_features,
                        'cameras':database_cameras})

print('save to file: {}'.format(save_file))
