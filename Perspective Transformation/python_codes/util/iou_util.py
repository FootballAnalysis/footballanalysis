import cv2 as cv
import numpy as np

from .projective_camera import ProjectiveCamera

class IouUtil:
    @staticmethod
    def homography_warp(h, image, dst_size, background_color):
        """
        :param h:
        :param image:
        :param dst_size:
        :param background_color:
        :return:
        """
        assert h.shape == (3, 3)
        im_dst = cv.warpPerspective(image, h, dst_size, borderMode=cv.BORDER_CONSTANT, borderValue=background_color)
        return im_dst

    @staticmethod
    def template_to_image_homography_uot(camera, template_h=74, template_w=115):
        """
        Only for UofT soccer model
        camera: measured in meter
        template_h, template_w: measured in yard
        template is an image
        template original point at top left of image, every pixel is 3 inch

        :param camera: projective camera
        :param template_h:
        :param template_w:
        :return: a homography matrix from template to image
        """
        assert (template_h, template_w) == (74, 115)

        yard2meter = 0.9144

        # flip template in y direction
        m1 = np.asarray([[1, 0, 0],
                         [0, -1, template_h],
                         [0, 0, 1]])
        # scale
        m2 = np.asarray([[yard2meter, 0, 0],
                         [0, yard2meter, 0],
                         [0, 0, 1]])
        tempalte2world = m2 @ m1
        world2image = camera.get_homography()
        h = world2image @ tempalte2world
        return h

    @staticmethod
    def iou_on_template_uot(gt_h, pred_h, im_h=720, im_w=1280, template_h=74, template_w=115):
        im = np.ones((im_h, im_w, 3), dtype=np.uint8) * 255
        gt_mask = IouUtil.homography_warp(np.linalg.inv(gt_h), im, (template_w, template_h), (0))
        pred_mask = IouUtil.homography_warp(np.linalg.inv(pred_h), im, (template_w, template_h), (0))

        val_intersection = (gt_mask != 0) * (pred_mask != 0)
        val_union = (gt_mask != 0) + (pred_mask != 0)
        u = float(np.sum(val_union))
        if u <= 0:
            iou = 0
        else:
            iou = 1.0 * np.sum(val_intersection) / u
        return iou


def ut_homography_warp():
    camera_data = np.asarray([640, 360, 3081.976880,
                              1.746393, -0.321347, 0.266827,
                              52.816224, -54.753716, 19.960425])

    u, v, fl = camera_data[0:3]
    rod_rot = camera_data[3:6]
    cc = camera_data[6:9]

    camera = ProjectiveCamera(fl, u, v, cc, rod_rot)
    h = camera.get_homography()
    inv_h = np.linalg.inv(h)
    im = cv.imread(r"C:\Users\mostafa\Desktop\test\train2\A25.jpg")
    assert im is not None


    #homography_warp(h, image, dst_size, background_color)
    template_size = (115, 74)
    warped_im = IouUtil.homography_warp(inv_h, im, (115, 74), (0, 0, 0))
    cv.imwrite('warped_image.jpg', warped_im)
    print("writed")

def ut_template_to_image_homography_uot():
    camera_data = np.asarray([640, 360, 3081.976880,
                              1.746393, -0.321347, 0.266827,
                              52.816224, -54.753716, 19.960425])

    u, v, fl = camera_data[0:3]
    rod_rot = camera_data[3:6]
    cc = camera_data[6:9]

    camera = ProjectiveCamera(fl, u, v, cc, rod_rot)

    h = IouUtil.template_to_image_homography_uot(camera)
    print('homography is {}'.format(h))

def ut_iou_on_template_uot_1():
    h1 = np.asarray(([[9.08451931e+00, -5.31615622e+00,  8.38673068e+01],
     [2.95203351e-03,  5.52517432e-01,  8.30620388e+01],
    [-1.69828350e-03, - 2.94390308e-03, 3.94388114e-01]]))
    h2 = np.asarray(([[2.83628335e+03, -1.58358576e+03, 2.43777943e+04],
                      [3.84106817e+01, 2.12539914e+02,  2.42631136e+04],
                      [-4.54655559e-01, - 7.51486304e-01,1.17493517e+02]]))

    iou = IouUtil.iou_on_template_uot(h1, h2)
    print('{}'.format(iou))

def ut_iou_on_template_uot_2():
    h1 = np.asarray(([[4.00507552e+01,  1.21935050e+01, -3.82721376e+03],
                      [-6.99948565e-01,  1.55281386e+00,  4.43575038e+02],
                      [6.88083482e-03, -1.23378011e-02,  1.00000000e+00]]))
    h2 = np.asarray(([[4.01621885e+01,  1.56120602e+01, -3.90765553e+03],
                      [-1.43462664e+00,  2.09267670e+00,  5.39727302e+02],
                      [7.76998497e-03, -1.18067686e-02,  1.00000000e+00]]))

    iou = IouUtil.iou_on_template_uot(h1, h2)
    print('{}'.format(iou))

def ut_generate_grassland_mask():
    # An example of generate soft mask for grassland segmentation
    import scipy.io as sio

    index = 16 - 1   # image index from 1
    data = sio.loadmat(r'C:\Users\mostafa\Desktop\test\SCCvSD-master/data/UoT_soccer/train_val.mat')
    annotation = data['annotation']
    homo = annotation[0][index][1]  # ground truth homography

    # step 1: generate a 'hard' grass mask
    template_h = 74
    template_w = 115
    tempalte_im = np.ones((template_h, template_w, 1), dtype=np.uint8) * 255

    grass_mask = IouUtil.homography_warp(homo, tempalte_im, (1280, 720), (0))
    cv.imshow('grass mask', grass_mask)
    cv.waitKey(0)

    # step 2: generate a 'soft' grass mask
    dist_threshold = 30  # change this value to change mask boundary
    _, binary_im = cv.threshold(grass_mask, 10, 255, cv.THRESH_BINARY_INV)

    dist_im = cv.distanceTransform(binary_im, cv.DIST_L2, cv.DIST_MASK_PRECISE)

    dist_im[dist_im > dist_threshold] = dist_threshold
    soft_mask = 1.0 - dist_im / dist_threshold  # normalize to [0, 1]

    cv.imshow('soft mask', soft_mask)
    cv.waitKey(0)

    # step 3: soft mask on the original image
    stacked_mask = np.stack((soft_mask,) * 3, axis=-1)
    im = cv.imread(r'C:\Users\mostafa\Desktop\test\SCCvSD-master/data/16.jpg')
    soft_im = cv.multiply(stacked_mask, im.astype(np.float32)).astype(np.uint8)
    cv.imshow('soft masked image', soft_im)
    cv.waitKey(0)


if __name__ == '__main__':
    ut_homography_warp()
    #ut_template_to_image_homography_uot()
    #ut_iou_on_template_uot1()
    #ut_iou_on_template_uot_2()
    ut_generate_grassland_mask()
