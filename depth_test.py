import cv2
import numpy as np
from StereoVisualOdom import StereoVisualOdometry
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataset_number = '00'

# Initialize dictionaries to store the values
P0 = []
P1 = []

# Open the file and read line by line
config_file_path = 'data_odometry_gray/dataset/sequences/' + dataset_number + '/calib.txt'
with open(config_file_path, 'r') as file:
    for line in file:
        # Check if the line starts with 'P0' or 'P0'
        if line.startswith('P0'):
            # Extract the numbers from the line, skip the first element
            P0 = [float(num) for num in line.split()[1:]]  
        elif line.startswith('P1'):
            # Extract the numbers from the line, skip the first element
            P1 = [float(num) for num in line.split()[1:]]  

P0_matrix = np.array(P0).reshape(3, 4)
P1_matrix = np.array(P1).reshape(3, 4)

# Extract M1, M2, t1, t2
M0 = P0_matrix[:, :3]
t0 = P0_matrix[:, 3]
M1 = P1_matrix[:, :3]
t1 = P1_matrix[:, 3]

# Compute camera centers
C1 = -np.linalg.inv(M0).dot(t0)
C2 = -np.linalg.inv(M1).dot(t1)

# Focal length and principal point 
focal_length = P0[0]  
principal_point = (P0[2], P0[6])
baseline = np.linalg.norm(C2 - C1)
print(focal_length, principal_point, baseline)

# Initialize object
odom = StereoVisualOdometry(focal_length, principal_point, baseline)

# Initialize the initial pose as the identity matrix if starting from origin
current_pose = np.eye(4)

print('Analyzing images...')

# Select a single image pair to analyze
frame_number = 750
file_name = f"{frame_number:06d}.png"
file_path_0 = 'data_odometry_gray/dataset/sequences/' + dataset_number + '/image_0/' + file_name
file_path_1 = 'data_odometry_gray/dataset/sequences/' + dataset_number + '/image_1/' + file_name

img0 = cv2.imread(file_path_0)
img1 = cv2.imread(file_path_1)

if img0 is not None and img1 is not None:
    disparity = odom.compute_disparity_map(img0, img1)

    depth_map = odom.disparity_to_depth(disparity)
    plt.figure(figsize=(10, 7))
    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar()
    plt.title('Depth Map')
    plt.show()

    # Detect and match features between the two images
    points_now, points_prev = odom.detect_and_match_features(img0, img1)

    # Create a plot to show both images and the matches
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    # Show first image
    ax[0].imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Image at Time t-1')
    ax[0].axis('off')

    # Show second image
    ax[1].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Image at Time t')
    ax[1].axis('off')

    # Draw lines between matched points
    for (x_now, y_now), (x_prev, y_prev) in zip(points_now, points_prev):
        # Line from the point in the first image to the same point in the second image
        con = matplotlib.patches.ConnectionPatch(xyA=(x_now, y_now), xyB=(x_prev, y_prev),
                                                coordsA="data", coordsB="data",
                                                axesA=ax[1], axesB=ax[0], color="yellow")
        ax[1].add_artist(con)

    plt.show()


print('Completed photo analysis!')
