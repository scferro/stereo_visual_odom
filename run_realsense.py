import cv2
import pyrealsense2 as rs
import numpy as np
from StereoVisualOdom import StereoVisualOdometry
import matplotlib.pyplot as plt
import time


### INPUTS ###
feature_type = 'ORB'        # ORB or SIFT of SuperPoint
calculate_time = True


# Create a context object to manage devices
context = rs.context()

# Get a list of connected devices
device_list = context.query_devices()

if len(device_list) == 0:
    print("No devices connected")
else:
    # Select the first device
    device = device_list[0]

    # Access the depth sensor
    depth_sensor = device.first_depth_sensor()

    # Disable the IR projector
    depth_sensor.set_option(rs.option.emitter_enabled, 0)

    print("IR dot projector has been disabled")

# Configure the pipeline
pipeline = rs.pipeline()
config = rs.config()

# Disable the IR dot projector
config.disable_stream(rs.stream.depth)  # Disable depth stream to disable IR dot projector

config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

# Start the pipeline
pipeline.start(config)

# Get the intrinsic parameters of the IR streams
profiles = pipeline.get_active_profile().get_streams()
print(profiles)
for profile in profiles:
    intrinsics = profile.as_video_stream_profile().get_intrinsics()
    focal_length = intrinsics.fx
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    break

# Get the baseline (distance between camera centers)
baseline = 0.05

# Create the camera matrix
camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

# Initialize the StereoVisualOdometry class
vo = StereoVisualOdometry(focal_length, (cx, cy), baseline, camera_matrix, feature_type=feature_type)

processing_times = []

print("Processing frames...")
count = 0
timer_start = time.time()
while True:
    # Get the frames
    frames = pipeline.wait_for_frames()

    # Get the left infrared frame
    left_ir_frame = frames.get_infrared_frame(1)
    left_ir_image = np.asanyarray(left_ir_frame.get_data())

    # Get the right infrared frame
    right_ir_frame = frames.get_infrared_frame(2)
    right_ir_image = np.asanyarray(right_ir_frame.get_data())

    start_time = time.time()
    vo.process_frame(left_ir_image, right_ir_image)
    end_time = time.time()
    processing_time = end_time - start_time
    processing_times.append(processing_time)

    count += 1
    second_timer = time.time() - timer_start
    if second_timer > 1.0:
        fps = count/second_timer
        print(f"Frame Rate: {fps} frames per second.")
        count = 0
        second_timer = 0
        timer_start = time.time()

    # Display the frames
    cv2.imshow('Left IR', left_ir_image)
    # cv2.imshow('Right IR', right_ir_image)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the pipeline and release resources
pipeline.stop()
cv2.destroyAllWindows()

# Display trajectory
trajectory = vo.get_trajectory()
fig, ax = plt.subplots()
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

# Display processing time statistics
avg_time = np.mean(processing_times)
print(f"Average processing time: {avg_time}")

fig, ax = plt.subplots()
avg_time_list = [avg_time] * len(processing_times)
ax.plot(processing_times, 'b-', label='Processing Time per Frame')
ax.plot(avg_time_list, 'r-', label='Average Processing Time')
ax.set_xlabel('Frame Number')
ax.set_ylabel('Time (seconds)')
ax.set_ylim([0, 0.25])
ax.set_title('Processing Time per Frame')
ax.legend()

plt.show()