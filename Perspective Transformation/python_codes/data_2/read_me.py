"""
worldcup2014.mat
It is the playing ground model of soccer.

points: N * 2, points in the model, for example, line intersections
line_segment_index: N * 2, line segment start/end point index
grid_points: a group of 2D points uniformly sampled inside of the playing ground.
             It is used to approximate the area of the playing ground.
"""

"""
worldcup_dataset_camera_parameter.mat
This is the camera parameters of the world cup dataset.
cc: for camera center, a 3D world coordinate
cc_mean:
cc_std:
cc_min:
cc_max:

fl: for focal length, a float value
fl_mean:
fl_std:
fl_min:
fl_max:
"""

"""
worldcup_sampled_cameras.mat
It has about ~90K positive camera pairs
pivot_cameras: N * 9, pivot cameras used in training
positive_cameras: N * 9, positive camera with the pivot camera
positive_ious: N * 9, IoU of each pivot and positive pair
"""

"""
train_data_10k.mat
It has 10k pivot and positive images.
pivot_images: N x 1 x 180 x 320, [mean, std] = [0.0188 0.128] (after normalized to [0, 1])
positive_images: N x 1 x 180 x 320
"""

"""
database_camera_feature.mat
It has about ~90K (camera, feature) pairs
cameras: N x 9
features: N x 16, deep feature
"""

"""
testset_feature.mat
186 testing images in the World Cup dataset
Deep feature, edge image and distance image of the testing set
edge_distances: (180, 320, 1, 186)
edge_map: (720, 1280, 1, 186)
features: (16, 186)
"""

"""
UoT_soccer: annotation from "Sports Field Localization Via Deep Structured Models"
http://www.cs.toronto.edu/~namdar/
train_val.mat
test.mat
"""
