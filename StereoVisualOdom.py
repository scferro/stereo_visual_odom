import cv2
import numpy as np

class StereoVisualOdometry:
    def __init__(self, focal_length, principal_point, baseline):
        self.focal_length = focal_length
        self.pp = principal_point
        self.baseline = baseline
        self.max_depth = 50
        self.min_disparity = 2
        self.max_disparity = 65
        self.points3d_prev = None
        self.previous_pose = None

    def detect_and_match_features(self, img_now, img_prev):
        # Using ORB detector for demonstration
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img_now, None)
        kp2, des2 = orb.detectAndCompute(img_prev, None)

        # Create BFMatcher object and match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of good matches
        points_now = np.zeros((len(matches), 2), dtype=np.float32)
        points_prev = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points_now[i, :] = kp1[match.queryIdx].pt
            points_prev[i, :] = kp2[match.trainIdx].pt

        return points_now, points_prev

    def compute_disparity_map(self, img_left, img_right):
        # Parameters for stereo matching (can be tuned for better performance)
        window_size = 5
        num_disparities = 16 * window_size
        block_size = 8
        stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=2 * block_size ** 2,
            P2=64 * block_size ** 2,
            disp12MaxDiff=2,
            uniquenessRatio=2,
            speckleWindowSize=16,
            speckleRange=4,
            mode=cv2.STEREO_SGBM_MODE_SGBM
        )

        
        # Ensure images are in grayscale
        img_0 = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_1 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Normalize images
        img_0 = cv2.equalizeHist(img_0)
        img_1 = cv2.equalizeHist(img_1)

        disparity = stereo.compute(img_0, img_1).astype(np.float32) / 16
        
        # Filtering the array to include only values greater than 0
        disparity[disparity == 0.0] = self.min_disparity
        disparity[disparity == -1.0] = self.min_disparity
        disparity[disparity > self.max_disparity] = self.max_disparity

        disparity = cv2.medianBlur(disparity, ksize=5)  

        return disparity

    def disparity_to_depth(self, disparity):
    # Set a minimum disparity threshold to avoid division by zero or very small numbers
        numerator = self.focal_length * self.baseline
        depth = numerator / disparity
        depth[depth > self.max_depth] = self.max_depth
        return depth
    
    def calculate_pose_change(self, current_pose):
        if self.previous_pose is not None:
            R_prev, t_prev = self.previous_pose
            R_current, t_current = current_pose

            # Calculate the relative rotation and translation
            R_change = np.linalg.inv(R_current) @ R_prev
            t_change = np.linalg.inv(R_current) @ (t_prev - t_current)
        else:
            R_change = np.eye(3)
            t_change = np.zeros((3, 1))

        self.previous_pose = current_pose
        return R_change, t_change

    def estimate_current_pose(self, points2d_now, depth_map):
        points3d_now = [self.project_to_3d(pt, depth_map[int(pt[1]), int(pt[0])]) for pt in points2d_now]
        points3d_now = np.array(points3d_now, dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))
        camera_matrix = np.array([[self.focal_length, 0, self.pp[0]], [0, self.focal_length, self.pp[1]], [0, 0, 1]], dtype=np.float32)
        _, rvec, tvec, _ = cv2.solvePnPRansac(points3d_now, points2d_now, camera_matrix, dist_coeffs)
        R, _ = cv2.Rodrigues(rvec)
        current_pose = (R, tvec)

        self.points3d_prev = points3d_now
        return current_pose
    
    def project_to_3d(self, point2d, depth):
        x = (point2d[0] - self.pp[0]) * depth / self.focal_length
        y = (point2d[1] - self.pp[1]) * depth / self.focal_length
        return [x, y, depth]
