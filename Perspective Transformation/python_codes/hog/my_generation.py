import scipy.io as sio
import cv2 as cv 


image_file=r"C:\Users\mostafa\Desktop\test\pytorch-two-GAN-master\pytorch-two-GAN-master\results\soccer_seg_detection_pix2pix\test_latest\images\A25_fake_D.png"
edge_map=cv.imread(image_file)

save_file = r'C:\Users\mostafa\Desktop\test\SCCvSD-master\python\hog\my_testset_feature.mat'

sio.savemat(save_file, {'edge_map':edge_map})