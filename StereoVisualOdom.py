import cv2
import numpy as np

class StereoVisualOdometry:
    def __init__(self, focal_length, pp, baseline, camera_matrix):
        self.focal_length = focal_length
        self.pp = pp
        self.baseline = baseline
        self.camera_matrix = camera_matrix
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.prev_left_image = None
        self.prev_right_image = None
        self.prev_points_3D = None
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.trajectory = []
        self.max_depth = 50
        self.min_disparity = 2
        self.max_disparity = 65
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=11,
            P1=8 * 1 * 11**2,
            P2=32 * 1 * 11**2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def compute_disparity(self, img_left, img_right):
        disparity = self.stereo.compute(img_left, img_right).astype(np.float32) / 16.0
        disparity[disparity == 0.0] = self.min_disparity
        disparity[disparity == -1.0] = self.min_disparity
        disparity[disparity > self.max_disparity] = self.max_disparity
        return cv2.medianBlur(disparity, ksize=5)
    
    def feature_matching(self, img1, img2):
        keypoints1, descriptors1 = self.sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(img2, None)
        if descriptors1 is None or descriptors2 is None:
            return []  # No descriptors to match

        matches = self.bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.25 * n.distance:
                if m.queryIdx < len(keypoints1) and m.trainIdx < len(keypoints2):
                    good_matches.append((m, keypoints1[m.queryIdx], keypoints2[m.trainIdx]))
        return good_matches

    def process_frame(self, left_image, right_image):
        if self.prev_left_image is None:
            self.prev_left_image = left_image
            self.prev_right_image = right_image
            return np.eye(4)

        disparity = self.compute_disparity(cv2.equalizeHist(self.prev_left_image), cv2.equalizeHist(self.prev_right_image))
        matches = self.feature_matching(self.prev_left_image, left_image)

        points1 = np.float32([m[1].pt for m in matches])
        points2 = np.float32([m[2].pt for m in matches])

        valid_3D_points, valid_2D_points = self.calculate_3D_points(disparity, points1, points2)
        if len(valid_3D_points) >= 4:
            self.estimate_motion(valid_3D_points, valid_2D_points)

        self.prev_left_image = left_image
        self.prev_right_image = right_image

        return self.current_transformation_matrix()

    def calculate_3D_points(self, disparity, points1, points2):
        valid_3D_points = []
        valid_2D_points = []
        for pt1, pt2 in zip(points1, points2):
            d = disparity[int(pt1[1]), int(pt1[0])]
            if d > 0:
                z = self.focal_length * self.baseline / d
                x = (pt1[0] - self.pp[0]) * z / self.focal_length
                y = (pt1[1] - self.pp[1]) * z / self.focal_length
                valid_3D_points.append([x, y, z])
                valid_2D_points.append(pt2)
        return np.array(valid_3D_points), np.array(valid_2D_points)

    def estimate_motion(self, points_3D, points_2D):
        _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3D, points_2D, self.camera_matrix, None)
        R, _ = cv2.Rodrigues(rvec)
        self.R = self.R @ R
        self.t += self.R @ tvec
        self.trajectory.append(self.t.ravel().copy())

    def current_transformation_matrix(self):
        return np.vstack((np.hstack((self.R, self.t)), [0, 0, 0, 1]))

    def get_pose(self):
        return self.R, self.t

    def get_trajectory(self):
        return np.array(self.trajectory)
