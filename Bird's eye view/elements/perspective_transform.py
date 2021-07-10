import pyflann 
import scipy.io as sio
import numpy as np
import cv2

from perspective_transform.util.synthetic_util import SyntheticUtil
from perspective_transform.util.iou_util import IouUtil
from perspective_transform.util.projective_camera import ProjectiveCamera
from arguments import Arguments
from perspective_transform.models.models import create_model

from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from perspective_transform.deep.siamese import BranchNetwork, SiameseNetwork
from perspective_transform.deep.camera_dataset import CameraDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Perspective_Transform():
    def __init__(self):
        self.query_index = 0
        self.current_directory = os.path.join(os.getcwd(), 'weights') 

        # Deep features
        deep_database_directory = os.path.join(self.current_directory, "data/deep/feature_camera_91k.mat")
        data=sio.loadmat(deep_database_directory)
        self.database_features = data['features']
        self.database_cameras = data['cameras']

        self.deep_model_directory =  os.path.join(self.current_directory, "deep_network.pth")
        self.net,self.data_transform = self.initialize_deep_feature()

        self.model =  self.initialize_two_GAN(self.current_directory)
        print('Perspective Transform model loaded!')


    def homography_matrix(self, image):
        image = cv2.resize(image,(1280,720)) # it shouldn't be changed
        edge_map ,seg_map = self.testing_two_GAN(image)
        
        # generate deep features
        test_features = self.generate_deep_feature(edge_map)

        # World Cup soccer template
        data = sio.loadmat(os.path.join(self.current_directory, "data/worldcup2014.mat"))
        model_points = data['points']
        model_line_index = data['line_segment_index']

        template_h = 74  # yard, soccer template
        template_w = 115
        
        # retrieve a camera using deep features
        flann = pyflann.FLANN()
        result, _ = flann.nn(self.database_features, test_features[self.query_index], 1, algorithm="kdtree", trees=8, checks=64)
        retrieved_index = result[0]

        """
        Retrieval camera: get the nearest-neighbor camera from database
        """
        retrieved_camera_data = self.database_cameras[retrieved_index]

        u, v, fl = retrieved_camera_data[0:3]
        rod_rot = retrieved_camera_data[3:6]
        cc = retrieved_camera_data[6:9]

        retrieved_camera = ProjectiveCamera(fl, u, v, cc, rod_rot)

        retrieved_h = IouUtil.template_to_image_homography_uot(retrieved_camera, template_h, template_w)

        retrieved_image = SyntheticUtil.camera_to_edge_image(retrieved_camera_data, model_points, model_line_index,
                                                    im_h=720, im_w=1280, line_width=2)

        
        """
        Refine camera: refine camera pose using Lucas-Kanade algorithm 
        """
        dist_threshold = 50
        query_dist = SyntheticUtil.distance_transform(edge_map)
        retrieved_dist = SyntheticUtil.distance_transform(retrieved_image)

        query_dist[query_dist > dist_threshold] = dist_threshold
        retrieved_dist[retrieved_dist > dist_threshold] = dist_threshold

        h_retrieved_to_query = SyntheticUtil.find_transform(retrieved_dist, query_dist)

        refined_h = h_retrieved_to_query@retrieved_h
        Warp_img = cv2.warpPerspective(seg_map, np.linalg.inv(refined_h), (115,74), borderMode=cv2.BORDER_CONSTANT)

        return np.linalg.inv(refined_h), Warp_img


    def initialize_deep_feature(self):
        
        # 2: load network
        branch = BranchNetwork()
        net = SiameseNetwork(branch)

        if os.path.isfile(self.deep_model_directory):
            checkpoint = torch.load(self.deep_model_directory, map_location=lambda storage, loc: storage)
            net.load_state_dict(checkpoint['state_dict'])
            
            # 3: setup computation device
        if torch.cuda.is_available():
            net = net.to(device)
            cudnn.benchmark = True
        
        
        normalize = transforms.Normalize(mean=[0.0188],
                                        std=[0.128])

        data_transform = transforms.Compose(
            [  transforms.ToTensor(),
            normalize,
            ]
        )
    
        return net,data_transform
    
    
    def generate_deep_feature(self, edge_map):
        """
        Extract feature from a siamese network
        input: network and edge images
        output: feature and camera
        """
        #parameters
        batch_size = 1
        model_name = self.deep_model_directory
        
        normalize = transforms.Normalize(mean=[0.0188],
                                        std=[0.128])

        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            normalize,
            ]
        )

        #resize image
        pivot_image = edge_map
        pivot_image = cv2.resize(pivot_image ,(320,180))

        pivot_image = cv2.cvtColor(pivot_image, cv2.COLOR_RGB2GRAY)
        pivot_image=np.reshape(pivot_image,(1,pivot_image.shape[0],pivot_image.shape[1]))
        # Note: assume input image resolution is 180 x 320 (h x w)

        data_loader = CameraDataset(pivot_image,
                                pivot_image,
                                batch_size,
                                -1,
                                data_transform,
                                is_train=False)

        # 2: load network
        branch = BranchNetwork()
        net = SiameseNetwork(branch)

        if os.path.isfile(model_name):
            checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
            net.load_state_dict(checkpoint['state_dict'])
        else:
            print('Error: file not found at {}'.format(model_name))
            
        # 3: setup computation device
        if torch.cuda.is_available():
            net = net.to(device)
            cudnn.benchmark = True

        features = []

        with torch.no_grad():
            for i in range(len(data_loader)):
                x, _ = data_loader[i]
                x = x.to(device)
                feat = net.feature_numpy(x) # N x C
                features.append(feat)
                # append to the feature list


        features = np.vstack((features))

        return features


    def testing_two_GAN(self, image):

        image=Image.fromarray(image)
        osize = [256,256]
        cropsize = osize
        image=transforms.Compose([transforms.Resize(osize, transforms.InterpolationMode.BICUBIC),
                                transforms.RandomCrop(cropsize),transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])(image)
        image=image.unsqueeze(0)        
            
        self.model.set_input(image)        
        self.model.test()
                    
        visuals = self.model.get_current_visuals()
        
        edge_map=visuals['fake_D'] 
        seg_map = visuals['fake_C']
        
        edge_map= cv2.resize(edge_map,(1280,720))
        seg_map= cv2.resize(seg_map,(1280,720))

        return edge_map , seg_map 

    def initialize_two_GAN(self, directory):
        opt = Arguments().parse()
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.continue_train = False

        self.model = create_model(opt)
        return self.model