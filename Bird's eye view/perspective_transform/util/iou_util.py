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
