import cv2
import numpy as np
from StereoVisualOdom import StereoVisualOdometry  # Import the provided class

def load_stereo_images(dataset_number, frame_number):
    base_path = '/home/scferro/Documents/msai495/cv_final_project/data_odometry_gray/dataset/sequences/'
    file_name = f"{frame_number:06d}.png"
    file_path_0 = f'{base_path}{dataset_number}/image_0/{file_name}'
    file_path_1 = f'{base_path}{dataset_number}/image_1/{file_name}'

    img0 = cv2.imread(file_path_0, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(file_path_1, cv2.IMREAD_GRAYSCALE)

    if img0 is None or img1 is None:
        print(f"Warning: Missing images at frame {frame_number}")
    return [img0, img1]

# INPUTS
dataset_number = '00'  
frame_number = 100

# Initialize to store the values
P0 = []
P1 = []

# Open the file and read line by line
config_file_path = f'/home/scferro/Documents/msai495/cv_final_project/data_odometry_gray/dataset/sequences/{dataset_number}/calib.txt'
with open(config_file_path, 'r') as file:
    for line in file:
        if line.startswith('P0'):
            P0 = [float(num) for num in line.split()[1:]]
        elif line.startswith('P1'):
            P1 = [float(num) for num in line.split()[1:]]

P0_matrix = np.array(P0).reshape(3, 4)
P1_matrix = np.array(P1).reshape(3, 4)

# Extract M0, M1, t0, t1
M0 = P0_matrix[:, :3]
t0 = P0_matrix[:, 3]
M1 = P1_matrix[:, :3]
t1 = P1_matrix[:, 3]

# Compute camera centers
C1 = -np.linalg.inv(M0).dot(t0)
C2 = -np.linalg.inv(M1).dot(t1)

# Focal length and principal point 
focal_length = P0[0]  
cx = P0[2]
cy = P0[6]
baseline = np.linalg.norm(C2 - C1)
camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

# Import images
print("Importing images...")
[img_left, img_right] = load_stereo_images(dataset_number, frame_number)
print("Import complete!")

if img_left is None or img_right is None:
    print("Error: Could not load the images.")
    exit(1)

# Initialize the StereoVisualOdometry object
vo = StereoVisualOdometry(focal_length, (cx, cy), baseline, camera_matrix, feature_type='SuperPoint')

# Detect features in the left image using SuperPoint
if vo.feature_type == 'SuperPoint':
    keypoints, descriptors = vo.superpoint_detect_and_compute(img_left)
else:
    keypoints, descriptors = vo.feature_detector.detectAndCompute(img_left, None)

# Adjust the size of the keypoints
large_keypoints = []
for kp in keypoints:
    kp.size = 10  # Increase the size of the keypoints
    large_keypoints.append(kp)

# Draw keypoints on the left image
img_with_keypoints = cv2.drawKeypoints(img_left, large_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the keypoints on the left image
cv2.imshow('Keypoints on Left Image', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
