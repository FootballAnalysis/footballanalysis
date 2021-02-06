import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

def main():
    
    model , visualizer = initialize()
    cap = cv2.VideoCapture(r'C:\Users\mostafa\Desktop\test\football7.mp4')
    edge_out = cv2.VideoWriter(r'C:\Users\mostafa\Desktop\GAN\billboard_borders.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 1, (256,256))
    
    while(cap.isOpened()):
        for i in range(int(cap.get(5))): ##just every second
            ret, frame = cap.read()
        
        #if cap.get(3)!= 1280.0 or  cap.get(4)!= 720.0 :
        #    frame = cv2.resize(frame,(1280,720))# ===> for videos which resolutions are greater 
        cv2.waitKey(1)
        
    

        # test
        image=Image.fromarray(frame)
        osize = [286,286]
        cropsize =[256,256]
        
        image=transforms.Compose([transforms.Scale(osize, Image.BICUBIC),transforms.RandomCrop(cropsize),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])(image)
        image=image.unsqueeze(0)        
            
        model.set_input(image)        
        model.test()
        
        visuals = model.get_current_visuals()
        GAN_result = visuals['fake_B']
        
        edge_out.write(GAN_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    edge_out.release()

def initialize():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.continue_train = False

    #opt.results_dir = 
    #opt.dataroot = 
    opt.name = r"C:\Users\mostafa\Desktop\GAN\checkpoints\datasets"
    #opt.checkpoints_dir =

    model = create_model(opt)
    visualizer = Visualizer(opt)
    return model , visualizer
    
    


if __name__ == '__main__':
    main()
