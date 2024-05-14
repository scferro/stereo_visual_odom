import cv2
import numpy as np
import matplotlib.pyplot as plt
from StereoVisualOdom import StereoVisualOdometry
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def load_stereo_images(dataset_number, num_images):
    images = []
    base_path = 'data_odometry_gray/dataset/sequences/'
    for frame_number in range(num_images):
        file_name = f"{frame_number:06d}.png"
        file_path_0 = f'{base_path}{dataset_number}/image_0/{file_name}'
        file_path_1 = f'{base_path}{dataset_number}/image_1/{file_name}'

        img0 = cv2.imread(file_path_0, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(file_path_1, cv2.IMREAD_GRAYSCALE)

        if img0 is not None and img1 is not None:
            images.append((img0, img1))
            if (frame_number+1) % 50 == 0:
                print("Imported " + str(frame_number+1) + " frames.")
        else:
            print(f"Warning: Missing images at frame {frame_number}")
            return images
    return images



dataset_number = '00'  
num_images = 500

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

# Import ground truth
print('Importing and plotting ground truth..')
gt_file = f'data_odometry_poses/dataset/poses/{dataset_number}.txt'
data = np.loadtxt(gt_file)
gt_x = data[:num_images, [11]]
gt_y = data[:num_images, [3]]

# Display gt trajectory
fig, ax = plt.subplots()
ax.plot(gt_x, gt_y, 'g-')
print('Ground truth imported!')

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
cx = P0[2]
cy = P0[6]
baseline = np.linalg.norm(C2 - C1)
camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

vo = StereoVisualOdometry(focal_length, (cx, cy), baseline, camera_matrix)

# Import images
print("Importing images...")
stereo_images = load_stereo_images(dataset_number, num_images)
print("Import complete!")

print("Processing frames...")
count = 0
for left_img, right_img in stereo_images:
    vo.process_frame(left_img, right_img)
    count += 1
    if count % 50 == 0:
        print(str(count) + " frames processed.")
        tf = vo.get_pose()
        # print(tf[1][2], tf[1][0])

# Display trajectory
trajectory = vo.get_trajectory()
ax.plot(-trajectory[:, 2], trajectory[:, 0], 'b-')  

# Mark the start and end points
start = trajectory[0]
end = trajectory[-1]
ax.scatter(start[2], start[1], color='red', s=30, zorder=5, label='Start')

ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('Camera Trajectory')
ax.legend()
ax.axis('equal')

plt.show()