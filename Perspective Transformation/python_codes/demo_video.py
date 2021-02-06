import pyflann 
import scipy.io as sio
import numpy as np
import cv2 as cv


from util.synthetic_util import SyntheticUtil
from util.iou_util import IouUtil
from util.projective_camera import ProjectiveCamera
from util.iou_util import ut_homography_warp
from utils import mouse_handler
from utils import get_two_points


from options.test_options import TestOptions
from models.models import create_model


import os
import argparse
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import time
from PIL import Image


from deep.siamese import BranchNetwork, SiameseNetwork
from deep.camera_dataset import CameraDataset


def generate_HOG_feature(edge_map):
    # Generate HoG feature from testset edge images

    # HoG parameters
    win_size = (128, 128)
    block_size = (32, 32)
    block_stride = (32, 32)
    cell_size = (32, 32)
    n_bins = 9

    im_h, im_w = 180, 320

    hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)

    h, w, c = edge_map.shape
    #n number of test images
    features = []
    
    edge_image = edge_map[:,:,:]
    edge_image = cv.resize(edge_image, (im_w, im_h))
    edge_image = cv.cvtColor(edge_image, cv.COLOR_BGR2GRAY)
    feat = hog.compute(edge_image)
    features.append(feat)

    features = np.squeeze(np.asarray(features), axis=2)


    return features

def initialize_deep_feature(deep_model_directory):
    
    cuda_id = -1 #use -1 for CPU and 0 for GPU 
    # 2: load network
    branch = BranchNetwork()
    net = SiameseNetwork(branch)

    if os.path.isfile(deep_model_directory):
        checkpoint = torch.load(deep_model_directory, map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['state_dict'])
        print('load model file from {}.'.format(deep_model_directory))
    else:
        print('Error: file not found at {}'.format(deep_model_directory))
        
        # 3: setup computation device
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(cuda_id))
        net = net.to(device)
        cudnn.benchmark = True
    print('computation device: {}'.format(device))

     
    
    normalize = transforms.Normalize(mean=[0.0188],
                                     std=[0.128])

    data_transform = transforms.Compose(
        [  transforms.ToTensor(),
        normalize,
        ]
    )

    
    return net,data_transform , device

def generate_deep_feature(edge_map,net,data_transform, device):
    """
    Extract feature from a siamese network
    input: network and edge images
    output: feature and camera
    """
    #parameters
    batch_size = 1
    
    #resize image
    pivot_image = edge_map
    

    pivot_image = cv.resize(pivot_image ,(320,180))
    
    
    
    pivot_image = cv.cvtColor(pivot_image, cv.COLOR_RGB2GRAY)
    
    pivot_images = np.reshape(pivot_image,(1,pivot_image.shape[0],pivot_image.shape[1]))
    
    print('Note: assume input image resolution is 180 x 320 (h x w)')

    data_loader = CameraDataset(pivot_images,
                            pivot_images,
                            batch_size,
                            -1,
                            data_transform,
                            is_train=False)
 
    features = []

    with torch.no_grad():
        for i in range(len(data_loader)):
            x, _ = data_loader[i]
            x = x.to(device)
            feat = net.feature_numpy(x) # N x C
            features.append(feat)
            # append to the feature list


    features = np.vstack((features))
    return features, pivot_image

def initialize_two_GAN(directory):
    opt = TestOptions().parse(directory)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.continue_train = False

    model = create_model(opt)
    return model

def testing_two_GAN(image, model):
    # test
      
    if __name__ == '__main__':
       
         
        image=Image.fromarray(image)
        osize = [512,256]

        cropsize = osize
        image=transforms.Compose([transforms.Scale(osize, Image.BICUBIC),transforms.RandomCrop(cropsize),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])(image)
        image=image.unsqueeze(0)        
            
        model.set_input(image)        
        model.test()
                    
        visuals = model.get_current_visuals() 
        edge_map= visuals['fake_D'] 
        seg_map = visuals['fake_C']
        edge_map = cv.resize(edge_map,(1280,720),interpolation=5)
        seg_map = cv.resize(seg_map,(1280,720))

        
        
            
    return edge_map,seg_map


################################### STEP 0 : get the addresses ###########################################
address_parser = argparse.ArgumentParser()
address_parser.add_argument('--image', required=True, type=str, help='sth like "./my_pic.png" ')
address_parser.add_argument('--advertising_image', required=False, type=str, help='sth like "./my_billboard.png" ')
        

address_args = address_parser.parse_args()

#########################################################################################################
        


################################## The base parameter which you have to set ############################################
feature_type ='deep'
print(feature_type)

#########################################################################################################################


query_index = 0

"""
Estimate an homogrpahy using edge images 
"""

######################## Step 1: get the current directory working:######################################################
from pathlib import Path

current_directory = str(Path(__file__).resolve().parent) # like this /home/skovorodkin/stack/scripts

print("current_directory is: "+ current_directory)




# ##################################################### Step 2: load data ###############################################
# database

if feature_type == "deep":
    deep_database_directory = current_directory + "/data_2/features/feature_camera_91k.mat"
    data=sio.loadmat(deep_database_directory)
    database_features = data['features']
    database_cameras = data['cameras']

    deep_model_directory = current_directory + "/deep/deep_network.pth"
    net,data_transform ,device= initialize_deep_feature(deep_model_directory)


else: #HOG feature database

    HOG_database_directory = current_directory + "/data_2/features/database_camera_feature_HoG.mat"
    data = sio.loadmat(HOG_database_directory)
    database_features=data['features']
    database_cameras = data['cameras']
    
#-------------------------------------------------------------------------------------------------------------
# testing edge image from two-GAN
model = initialize_two_GAN(current_directory)


cap = cv.VideoCapture(address_args.image)

warped_out = cv.VideoWriter(current_directory + r"/warped_output.avi",cv.VideoWriter_fourcc('M','J','P','G'), 1, (460,296))
retrieved_out = cv.VideoWriter(current_directory + r"/retrieved_output.avi",cv.VideoWriter_fourcc('M','J','P','G'), 1, (1280,720))

if address_args.advertising_image :
    overlayed_out = cv.VideoWriter(current_directory + r"/overlayed_output.avi",cv.VideoWriter_fourcc('M','J','P','G'), 1, (1280,720))



#ret1=cap.set(cv.CAP_PROP_FRAME_WIDTH,1024)
#ret2=cap.set(cv.CAP_PROP_FRAME_HEIGHT,720)
print(cap.get(3))
print(cap.get(4))
print(cap.get(7))

while(cap.isOpened()):
    start_time = time.time() ## ===> for measuring execution time

    for i in range(int(cap.get(5))): ##just every second
        ret, frame = cap.read()
        
    
    if cap.get(3)!= 1280.0 or  cap.get(4)!= 720.0 :
        frame = cv.resize(frame,(1280,720))# ===> for videos which resolutions are greater 
    cv.waitKey(1)
    
    edge_map ,seg_map = testing_two_GAN (frame,model)
    
    
    
    

    ########################################################################################
    if feature_type == "deep":
        test_features , reduced_edge_map = generate_deep_feature(edge_map ,net,data_transform,device)

    else: #HOG feature
        test_features = generate_HOG_feature(edge_map)

    #--------------------------------------------------------------------------------------------------------    
    # World Cup soccer template
    data = sio.loadmat(current_directory + "/data_2/worldcup2014.mat")
    model_points = data['points']
    model_line_index = data['line_segment_index']

    template_h = 74  # yard, soccer template
    template_w = 115

    ##########################################################################################################################################
    ############################################ Step 2: retrieve a camera using deep features ################################################
    flann = pyflann.FLANN()
    result, _ = flann.nn(database_features, test_features[query_index], 1, algorithm="kdtree", trees=8, checks=64)
    retrieved_index = result[0]
    

    """
    Retrieval camera: get the nearest-neighbor camera from database
    """
    retrieved_camera_data = database_cameras[retrieved_index]

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
    ####################################################################################################################

    ## Warp source image to destination based on homography
    im_out = cv.warpPerspective(seg_map, np.linalg.inv(refined_h), (115,74), borderMode=cv.BORDER_CONSTANT)

    frame = cv.resize(frame,(1280,720),interpolation = cv.INTER_CUBIC)

    ################################################### advertisement overlaying #######################################
    if address_args.advertising_image :
        billboard = cv.imread(address_args.advertising_image)
        billboard= np.tile(billboard, (1,2, 1))
        billboard = cv.resize(billboard,(115,74),interpolation = cv.INTER_CUBIC)
        
        im_out_2 = cv.warpPerspective(billboard,refined_h, (1280,720), borderMode=cv.BORDER_CONSTANT)
        
        
        im_out_2 = cv.addWeighted(frame,0.9,im_out_2,0.2,0.0)
        #cv.imshow("ss",im_out_2 )
        #cv.waitKey()


    ###################################################################################################################
    
    
    
    model_address=current_directory + "/model.jpg"
    model_image=cv.imread(model_address)
    model_image=cv.resize(model_image,(115,74))

    new_image=cv.addWeighted(model_image,1,im_out,1,0)

    new_image=cv.resize(new_image,(460,296),interpolation=1)
    
    
    # Display images
    """cv.waitKey(200)
    cv.imshow('frame',frame)
    cv.waitKey()
    cv.imshow('overlayed image', im_out_2)
    cv.waitKey()
    cv.imshow('Edge image of retrieved camera', retrieved_image)
    cv.waitKey()
    cv.imshow("Warped Source Image", new_image)
    cv.waitKey()"""
    
    if address_args.advertising_image :
        overlayed_out.write(im_out_2)
    retrieved_out.write(retrieved_image)
    warped_out.write(new_image)
    
    

    print("--- %s seconds ---" % (time.time() - start_time))




    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

warped_out.release()
retrieved_out.release()

if address_args.advertising_image :
    overlayed_out.release()

cv.destroyAllWindows()











