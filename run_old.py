import cv2
import numpy as np
from StereoVisualOdom_old import StereoVisualOdometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Set the number of image pairs to analyze and the dataset to look at
num_images = 100
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

# Initialize a list to store the poses
pose_list = [np.array([0., 0., 0.])]  # Start at the origin

# Initialize rotation and translation
R_current = np.eye(3)
t_current = np.zeros((3, 1))

print('Analyzing images...')
img0_prev = None

# Simulate a process of reading stereo images and estimating poses
for frame_number in range(num_images):
    file_name = f"{frame_number:06d}.png" 
    file_path_0 = 'data_odometry_gray/dataset/sequences/' + dataset_number + '/image_0/' + file_name
    file_path_1 = 'data_odometry_gray/dataset/sequences/' + dataset_number + '/image_1/' + file_name

    img0 = cv2.imread(file_path_0)
    img1 = cv2.imread(file_path_1)

    if img0 is not None and img1 is not None and img0_prev is not None:
        points_now, points_prev = odom.detect_and_match_features(img0, img0_prev)
        disparity = odom.compute_disparity_map(img0, img1)
        # depth_map = odom.disparity_to_depth(disparity)
        depth_map = disparity
        current_pose = odom.estimate_current_pose(points_now, depth_map)
        
        R_change, t_change = odom.calculate_pose_change(current_pose)
        print(R_change, t_change)

        R_updated = R_change @ R_current 
        t_updated = R_change @ t_current + t_change 
        print(t_current.ravel())
        pose_list.append(t_current.ravel())

    img0_prev = img0

print('Completed photo analysis!')
print('Plotting calculated odometry...')

# Plotting the poses
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, z coordinates from pose_list
x_coords = []
y_coords = []
z_coords = []
for i in range(num_images):
    x_coords.append(pose_list[i][0])
    y_coords.append(pose_list[i][1])
    z_coords.append(pose_list[i][2])

# Plot the trajectory
ax.plot(x_coords, y_coords, z_coords, marker='o')

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Trajectory of the Camera')

print('Displaying odometry!')

# Display the plot
plt.show()