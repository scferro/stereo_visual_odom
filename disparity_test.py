import cv2
import numpy as np
import matplotlib.pyplot as plt
from StereoVisualOdom import StereoVisualOdometry

def compute_and_show_disparity(left_image_path, right_image_path):
    # Read images
    img_left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize dummy parameters (use actual calibrated values in practice)
    dummy_focal_length = 718.8560
    dummy_pp = (607.1928, 185.2157)
    dummy_baseline = 0.573
    dummy_camera_matrix = np.array([[dummy_focal_length, 0, dummy_pp[0]], [0, dummy_focal_length, dummy_pp[1]], [0, 0, 1]])

    # Create StereoVisualOdometry object
    vo = StereoVisualOdometry(dummy_focal_length, dummy_pp, dummy_baseline, dummy_camera_matrix)

    # Compute disparity
    disparity = vo.compute_disparity(img_left, img_right)
    
    # Display the disparity map
    plt.imshow(disparity, cmap='jet')
    plt.colorbar()
    plt.show()

base_path = 'data_odometry_gray/dataset/sequences/'
frame_number = 88
dataset_number = '03'
file_name = f"{frame_number:06d}.png"
file_path_0 = f'{base_path}{dataset_number}/image_0/{file_name}'
file_path_1 = f'{base_path}{dataset_number}/image_1/{file_name}'
compute_and_show_disparity(file_path_0, file_path_1)
