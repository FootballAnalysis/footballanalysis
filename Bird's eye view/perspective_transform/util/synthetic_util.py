import numpy as np
import cv2 as cv

from perspective_transform.util.rotation_util import RotationUtil
from perspective_transform.util.projective_camera import ProjectiveCamera

class SyntheticUtil:
    @staticmethod
    def camera_to_edge_image(camera_data,
                             model_points, model_line_segment,
                             im_h=720, im_w=1280, line_width=4):
        """
         #720,1280
         Project (line) model images using the camera
        :param camera_data: 9 numbers
        :param model_points:
        :param model_line_segment:
        :param im_h: 720
        :param im_w: 1280
        :return: H * W * 3 OpenCV image
        """
        assert camera_data.shape[0] == 9

        u, v, fl = camera_data[0:3]
        rod_rot = camera_data[3:6]
        cc = camera_data[6:9]

        camera = ProjectiveCamera(fl, u, v, cc, rod_rot)
        im = np.zeros((im_h, im_w, 3), dtype=np.uint8)
        n = model_line_segment.shape[0]
        color = (255,255,255)
        for i in range(n):
            idx1, idx2 = model_line_segment[i][0], model_line_segment[i][1]
            p1, p2 = model_points[idx1], model_points[idx2]
            q1 = camera.project_3d(p1[0], p1[1], 0.0, 1.0)
            q2 = camera.project_3d(p2[0], p2[1], 0.0, 1.0)
            q1 = np.rint(q1).astype(np.int)
            q2 = np.rint(q2).astype(np.int)
            cv.line(im, tuple(q1), tuple(q2), color, thickness=line_width)
        return im

    @staticmethod
    def distance_transform(img):
        """
        :param img: OpenCV Image
        :return:
        """
        h, w, c = img.shape
        if c == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            assert c == 1

        _, binary_im = cv.threshold(img, 10, 255, cv.THRESH_BINARY_INV)

        dist_im = cv.distanceTransform(binary_im, cv.DIST_L2, cv.DIST_MASK_PRECISE)
        return dist_im

    @staticmethod
    def find_transform(im_src, im_dst):
        warp = np.eye(3, dtype=np.float32)
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 0.001)
        try:
            (_, warp) = cv.findTransformECC(im_src, im_dst, warp, cv.MOTION_HOMOGRAPHY, criteria,None,1)
        except:
            print('Warning: find transform failed. Set warp as identity')
        return warp

    @staticmethod
    def generate_ptz_cameras(cc_statistics,
                             fl_statistics,
                             roll_statistics,
                             pan_range, tilt_range,
                             u, v,
                             camera_num):
        """
        Input: PTZ camera base information
        Output: randomly sampled camera parameters
        :param cc_statistics:
        :param fl_statistics:
        :param roll_statistics:
        :param pan_range:
        :param tilt_range:
        :param u:
        :param v:
        :param camera_num:
        :return: N * 9 cameras
        """
        cc_mean, cc_std, cc_min, cc_max = cc_statistics
        fl_mean, fl_std, fl_min, fl_max = fl_statistics
        roll_mean, roll_std, roll_min, roll_max = roll_statistics
        pan_min, pan_max = pan_range
        tilt_min, tilt_max = tilt_range

        # generate random values from the distribution
        camera_centers = np.random.normal(cc_mean, cc_std, (camera_num, 3))
        focal_lengths = np.random.normal(fl_mean, fl_std, (camera_num, 1))
        rolls = np.random.normal(roll_mean, roll_std, (camera_num, 1))
        pans = np.random.uniform(pan_min, pan_max, camera_num)
        tilts = np.random.uniform(tilt_min, tilt_max, camera_num)

        cameras = np.zeros((camera_num, 9))
        for i in range(camera_num):
            base_rotation = RotationUtil.rotate_y_axis(0) @ RotationUtil.rotate_x_axis(rolls[i]) @\
                RotationUtil.rotate_x_axis(-90)
            pan_tilt_rotation = RotationUtil.pan_y_tilt_x(pans[i], tilts[i])
            rotation = pan_tilt_rotation @ base_rotation
            rot_vec, _ = cv.Rodrigues(rotation)

            cameras[i][0], cameras[i][1] = u, v
            cameras[i][2] = focal_lengths[i]
            cameras[i][3], cameras[i][4], cameras[i][5] = rot_vec[0], rot_vec[1], rot_vec[2]
            cameras[i][6], cameras[i][7], cameras[i][8] = camera_centers[i][0], camera_centers[i][1], camera_centers[i][2]
        return cameras

    @staticmethod
    def sample_positive_pair(pp, cc, base_roll, pan, tilt, fl,
                             pan_std, tilt_std, fl_std):
        """
        Sample a camera that has similar pan-tilt-zoom with (pan, tilt, fl).
        The pair of the camera will be used a positive-pair in the training
        :param pp: [u, v]
        :param cc: camera center
        :param base_roll: camera base, roll angle
        :param pan:
        :param tilt:
        :param fl:
        :param pan_std:
        :param tilt_std:
        :param fl_std:
        :return:
        """

        def get_nearby_data(d, std):
            assert std > 0
            delta = np.random.uniform(-0.5, 0.5, 1) * std
            return d + delta

        pan = get_nearby_data(pan, pan_std)
        tilt = get_nearby_data(tilt, tilt_std)
        fl = get_nearby_data(fl, fl_std)

        camera = np.zeros(9)
        camera[0] = pp[0]
        camera[1] = pp[1]
        camera[2] = fl

        base_rotation = RotationUtil.rotate_y_axis(0) @ RotationUtil.rotate_y_axis(base_roll) @\
                        RotationUtil.rotate_x_axis(-90)
        pan_tilt_rotation = RotationUtil.pan_y_tilt_x(pan, tilt)
        rotation = pan_tilt_rotation @ base_rotation
        rot_vec = RotationUtil.rotation_matrix_to_Rodrigues(rotation)
        camera[3: 6] = rot_vec.squeeze()
        camera[6: 9] = cc
        return camera

    @staticmethod
    def generate_database_images(pivot_cameras, positive_cameras,
                                 model_points, model_line_segment):

        """
        Default size 180 x 320 (h x w)
        Generate database image for siamese network training
        :param pivot_cameras:
        :param positive_cameras:
        :return:
        """
        n = pivot_cameras.shape[0]
        assert n == positive_cameras.shape[0]

        # N x 1 x H x W pivot images
        # N x 1 x H x w positive image
        # negative pairs are randomly selected
        im_h, im_w = 180, 320
        pivot_images = np.zeros((n, 1, im_h, im_w), dtype=np.uint8)
        positive_images = np.zeros((n, 1, im_h, im_w), dtype=np.uint8)


        for i in range(n):
            piv_cam = pivot_cameras[i,:]
            pos_cam = positive_cameras[i,:]

            piv_im = SyntheticUtil.camera_to_edge_image(piv_cam, model_points, model_line_segment, 720, 1280, 4)
            pos_im = SyntheticUtil.camera_to_edge_image(pos_cam, model_points, model_line_segment, 720, 1280, 4)

            # to a smaller image
            piv_im = cv.resize(piv_im, (im_w, im_h))
            pos_im = cv.resize(pos_im, (im_w, im_h))

            # to a gray image
            piv_im = cv.cvtColor(piv_im, cv.COLOR_BGR2GRAY)
            pos_im = cv.cvtColor(pos_im, cv.COLOR_RGB2GRAY)

            pivot_images[i, 0,:,:] = piv_im
            positive_images[i, 0, :,:] = pos_im

        return (pivot_images, positive_images)
