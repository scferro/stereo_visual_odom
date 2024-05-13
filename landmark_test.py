import cv2
import numpy as np
import matplotlib.pyplot as plt
from StereoVisualOdom import StereoVisualOdometry

def display_landmark_points(image1_path, image2_path):
    # Read images in color for display
    img1_color = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    img2_color = cv2.imread(image2_path, cv2.IMREAD_COLOR)
    
    # Convert images to grayscale for processing
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    if img1_gray is None or img2_gray is None or img1_color is None or img2_color is None:
        print("Error loading images. Please check the paths.")
        return

    # Initialize dummy parameters
    dummy_focal_length = 718.8560
    dummy_pp = (607.1928, 185.2157)
    dummy_baseline = 0.573
    dummy_camera_matrix = np.array([[dummy_focal_length, 0, dummy_pp[0]], [0, dummy_focal_length, dummy_pp[1]], [0, 0, 1]])

    # Create StereoVisualOdometry object
    vo = StereoVisualOdometry(dummy_focal_length, dummy_pp, dummy_baseline, dummy_camera_matrix)

    # Detect features and matches using grayscale images
    matches = vo.feature_matching(img1_gray, img2_gray)

    # Ensure there are matches to draw
    if not matches:
        print("No matches found. Unable to draw.")
        return

    # Create a new image that concatenates the two color images horizontally
    h1, w1 = img1_color.shape[:2]
    h2, w2 = img2_color.shape[:2]
    height = max(h1, h2)
    width = w1 + w2
    matched_image = np.zeros((height, width, 3), dtype='uint8')
    matched_image[:h1, :w1, :] = img1_color
    matched_image[:h2, w1:w1 + w2, :] = img2_color
    empty_image = matched_image.copy()

    # Define colors for drawing
    color_line = (0, 255, 0)  # Green lines
    color_keypoints = (0, 0, 255)  # Red keypoints

    # Draw lines between matched points
    for match, kp1, kp2 in matches:
        # Draw the keypoints in the images
        x1, y1 = int(kp1.pt[0]), int(kp1.pt[1])
        x2, y2 = int(kp2.pt[0]) + w1, int(kp2.pt[1])  # Note the shift for the x-coordinate of the second image
        cv2.line(matched_image, (x1, y1), (x2, y2), color_line, 1)
        cv2.circle(matched_image, (x1, y1), 5, color_keypoints, 1)
        cv2.circle(matched_image, (x2, y2), 5, color_keypoints, 1)

    # Display the matches
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(empty_image, cv2.COLOR_BGR2RGB))
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.show()

# Specify the paths to your sequential images
base_path = 'data_odometry_gray/dataset/sequences/'
frame_number = 88
dataset_number = '03'
file_name_0 = f"{frame_number:06d}.png"
file_name_1 = f"{(frame_number+1):06d}.png"
file_path_0 = f'{base_path}{dataset_number}/image_0/{file_name_0}'
file_path_1 = f'{base_path}{dataset_number}/image_0/{file_name_1}'

display_landmark_points(file_path_0, file_path_1)
