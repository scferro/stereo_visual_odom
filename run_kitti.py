import cv2
import numpy as np
import matplotlib.pyplot as plt
from StereoVisualOdom import StereoVisualOdometry
import time
import matplotlib.animation as animation

def load_stereo_images(dataset_number, num_images):
    images = []
    base_path = '/home/scferro/Documents/msai495/cv_final_project/data_odometry_gray/dataset/sequences/'
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


### INPUTS ###
dataset_number = '00'  
num_images = 4500
feature_type = 'SIFT'        # ORB or SIFT of SuperPoint
calculate_time = True
calculate_error = True


# Initialize to store the values
P0 = []
P1 = []

# Open the file and read line by line
config_file_path = '/home/scferro/Documents/msai495/cv_final_project/data_odometry_gray/dataset/sequences/' + dataset_number + '/calib.txt'
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
gt_file = f'/home/scferro/Documents/msai495/cv_final_project/data_odometry_poses/dataset/poses/{dataset_number}.txt'
data = np.loadtxt(gt_file)
gt_x = data[:num_images, [11]]
gt_y = data[:num_images, [3]]
gt_z = data[:num_images, [7]]

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

vo = StereoVisualOdometry(focal_length, (cx, cy), baseline, camera_matrix, feature_type=feature_type)

# Import images
print("Importing images...")
stereo_images = load_stereo_images(dataset_number, num_images)
print("Import complete!")

if calculate_time:
    processing_times = []
if calculate_error:
    errors = []

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'output_video_{feature_type}_{dataset_number}.mp4', fourcc, 30.0, (1280, 480))

print("Processing frames...")
count = 0
for left_img, right_img in stereo_images:
    if calculate_time:
        start_time = time.time()
    vo.process_frame(left_img, right_img)
    if calculate_time:
        end_time = time.time()
        processing_times.append(end_time - start_time)
    count += 1
    if count % 50 == 0:
        print(str(count) + " frames processed.")
        tf = vo.get_pose()

    if calculate_error:
        estimated_pose = vo.get_pose()
        x_est = -estimated_pose[1][2]
        y_est = estimated_pose[1][0]
        z_est = -estimated_pose[1][1]
        x_gt = gt_x[count-1]
        y_gt = gt_y[count-1]
        z_gt = gt_z[count-1]
        error = np.sqrt((x_est-x_gt)**2 + (y_est-y_gt)**2 + (z_est-z_gt)**2)
        errors.append(error)
    
    # Create a side-by-side image for video
    combined_img = np.hstack((left_img, right_img))
    combined_img_color = cv2.cvtColor(combined_img, cv2.COLOR_GRAY2BGR)
    out.write(combined_img_color)

out.release()

# Display trajectory
trajectory = vo.get_trajectory()
fig, ax = plt.subplots()
ax.plot(gt_x, gt_y, 'g-', label='Ground Truth')
ax.plot(-trajectory[:, 2], trajectory[:, 0], 'b-', label='Estimated')

# Mark the start and end points
start = trajectory[0]
end = trajectory[-1]
ax.scatter(start[2], start[0], color='red', s=30, zorder=5, label='Start')

ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title(f'Camera Trajectory - {feature_type} - Dataset {dataset_number}')
ax.legend()
ax.axis('equal')

plt.show()

if calculate_time:
    fig, ax = plt.subplots()
    avg_time = np.mean(processing_times)
    avg_time_list = [avg_time] * len(processing_times)
    ax.plot(processing_times, 'b-', label='Processing Time per Frame')
    ax.plot(avg_time_list, 'r-', label='Average Processing Time')
    ax.set_xlabel('Frame Number') 
    ax.set_ylabel('Time (seconds)') 
    ax.set_ylim([0, 0.25])  
    ax.set_title(f'Processing Time per Frame - {feature_type} - Dataset {dataset_number}')  
    ax.legend()
    print(f"Average processing time: {avg_time}")

    plt.show()

if calculate_error:
    fig, ax = plt.subplots()
    ax.plot(errors, 'b-', label='Position Error per Frame')
    ax.set_xlabel('Frame Number')  
    ax.set_ylabel('Error (meters)')  
    ax.set_title(f'Position Error - {feature_type} - Dataset {dataset_number}') 
    ax.legend()

    plt.show()

# Animation
fig, ax = plt.subplots(figsize=(10, 8))
line_gt, = ax.plot([], [], 'g-', lw=2, label='Ground Truth')
line_est, = ax.plot([], [], 'b-', lw=2, label='Estimated')
point, = ax.plot([], [], 'ro', ms=5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'Camera Trajectory Animation - {feature_type} - Dataset {dataset_number}')
ax.legend()
ax.axis('equal')

x_min, x_max = min(min(gt_x), -max(trajectory[:, 2])), max(max(gt_x), -min(trajectory[:, 2]))
y_min, y_max = min(min(gt_y), min(trajectory[:, 0])), max(max(gt_y), max(trajectory[:, 0]))

# Increase the margin to 20% of the range
margin = 0.2 * max(x_max - x_min, y_max - y_min)
ax.set_xlim(x_min - margin, x_max + margin)
ax.set_ylim(y_min - margin, y_max + margin)

def init():
    line_gt.set_data([], [])
    line_est.set_data([], [])
    point.set_data([], [])
    return line_gt, line_est, point

def animate(i):
    line_gt.set_data(gt_x[:i+1], gt_y[:i+1])
    x = -trajectory[:i+1, 2]
    y = trajectory[:i+1, 0]
    line_est.set_data(x, y)
    point.set_data(x[-1], y[-1])
    return line_gt, line_est, point

# Calculate the desired frame rate (you can adjust this as needed)
fps = 30

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(trajectory), interval=1000/fps, blit=True, repeat=False)

# Save the animation as a video file
Writer = animation.FFMpegWriter(fps=fps)
ani.save(f'trajectory_animation_{feature_type}_{dataset_number}.mp4', writer=Writer)

plt.close(fig)  # Close the figure after saving

print(f"Trajectory animation saved as trajectory_animation_{feature_type}_{dataset_number}.mp4")