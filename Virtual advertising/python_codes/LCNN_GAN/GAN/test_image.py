import os
from options.test_options import TestOptions
from models.models import create_model
from util.visualizer import Visualizer





def main():
    
    model , visualizer = initialize()
    

    # test
    image=Image.fromarray(image)
    osize = [286,286]
    cropsize =[256,256]
    
    image=transforms.Compose([transforms.Scale(osize, Image.BICUBIC),transforms.RandomCrop(cropsize),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])(image)
    image=image.unsqueeze(0)        
        
    model.set_input(image)        
    model.test()
    
    visuals = model.get_current_visuals()
    GAN_result = visuals['fake_B']
        

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
