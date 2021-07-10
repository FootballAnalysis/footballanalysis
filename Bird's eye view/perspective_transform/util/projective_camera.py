import numpy as np
import cv2 as cv

class ProjectiveCamera:
    def __init__(self, fl, u, v, cc, rod_rot):
        """
        :param fl:
        :param u:
        :param v:
        :param cc:
        :param rod_rot:
        """
        self.K = np.zeros((3, 3)) # calibration matrix
        self.camera_center = np.zeros(3)
        self.rotation = np.zeros(3)

        self.P = np.zeros((3, 4)) # projection matrix

        self.set_calibration(fl, u, v)
        self.set_camera_center(cc)
        self.set_rotation(rod_rot)

    def set_calibration(self, fl, u, v):
        """
        :param fl:
        :param u:
        :param v:
        :return:
        """
        self.K = np.asarray([[fl, 0, u],
                             [0, fl, v],
                             [0, 0, 1]])
        self._recompute_matrix()

    def set_camera_center(self, cc):
        assert cc.shape[0] == 3

        self.camera_center[0] = cc[0]
        self.camera_center[1] = cc[1]
        self.camera_center[2] = cc[2]
        self._recompute_matrix()

    def set_rotation(self, rod_rot):
        """
        :param rod_rot: Rodrigues vector
        :return:
        """
        assert rod_rot.shape[0] == 3

        self.rotation[0] = rod_rot[0]
        self.rotation[1] = rod_rot[1]
        self.rotation[2] = rod_rot[2]
        self._recompute_matrix()

    def project_3d(self, x, y, z, w=1.0):
        """
        :param x:
        :param y:
        :param z:
        :return:
        """
        p = np.zeros(4)
        p[0],p[1],p[2], p[3] = x, y, z, w
        q = self.P @ p
        assert q[2] != 0.0
        return (q[0]/q[2], q[1]/q[2])

    def get_homography(self):
        """
        homography matrix from the projection matrix
        :return:
        """
        h = self.P[:, [0, 1,3]]
        return h


    def _recompute_matrix(self):
        """
        :return:
        """
        P = np.zeros((3, 4))
        for i in range(3):
            P[i][i] = 1.0

        for i in range(3):
            P[i][3] = -self.camera_center[i]

        r, _ = cv.Rodrigues(self.rotation)
        #print(r)
        #print('{} {} {}'.format(self.K.shape, r.shape, P.shape))
        self.P = self.K @ r @ P

