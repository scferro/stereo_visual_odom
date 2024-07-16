import cv2
import pyrealsense2 as rs
import numpy as np
from StereoVisualOdom import StereoVisualOdometry
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

### INPUTS ###
feature_type = 'SuperPoint'        # ORB or SIFT or SuperPoint
calculate_fps = True

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

config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 60)

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

fps_list = []
out = cv2.VideoWriter('output_video_' + feature_type + '.mp4', cv2.VideoWriter_fourcc(*'XVID'), 60, (640, 480))

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

    vo.process_frame(left_ir_image, right_ir_image)

    count += 1
    second_timer = time.time() - timer_start
    if second_timer > 1.0 and calculate_fps:
        fps = count / second_timer
        print(f"Frame Rate: {fps} frames per second.")
        count = 0
        second_timer = 0
        timer_start = time.time()
        fps_list.append(fps)

    # Convert the left infrared frame to 3-channel RGB
    left_ir_image_rgb = cv2.cvtColor(left_ir_image, cv2.COLOR_GRAY2BGR)

    # Display the frames
    cv2.imshow('Left IR', left_ir_image)
    # cv2.imshow('Right IR', right_ir_image)

    out.write(left_ir_image_rgb)  # Write the frame to the video file

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


pipeline.stop()
out.release()  # Release the VideoWriter object
cv2.destroyAllWindows()

# Display trajectory
trajectory = vo.get_trajectory()
fig, ax = plt.subplots()
ax.plot(-trajectory[:, 2], trajectory[:, 0], 'b-')

# Mark the start and end points
start = trajectory[0]
end = trajectory[-1]
ax.scatter(start[2], start[0], color='red', s=30, zorder=5, label='Start')

ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('Camera Trajectory - ' + feature_type)
ax.legend()
ax.axis('equal')

dist = (trajectory[-1, 2]**2 + trajectory[-1, 0]**2)**0.5
print("Distance from Origin: " + str(dist))

plt.show()

if calculate_fps:
    # Display processing time statistics
    avg_time = np.mean(fps_list)

    fig, ax = plt.subplots()
    avg_time_list = [avg_time] * len(fps_list)
    ax.plot(fps_list, 'b-', label='Framerate')
    ax.plot(avg_time_list, 'r-', label='Average Framerate')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Framerate (fps)')
    # ax.set_ylim([0, ])
    ax.set_title('Image Processing Framerate - ' + feature_type)
    ax.legend()

    plt.show()

fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', lw=2)
point, = ax.plot([], [], 'ro', ms=5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Camera Trajectory Animation - ' + feature_type)
ax.axis('equal')

range_x = np.absolute(max(trajectory[:, 2]) - min(trajectory[:, 2]))
range_y = np.absolute(max(trajectory[:, 0]) - min(trajectory[:, 0]))

x_min = (-max(trajectory[:, 2])-(range_x*0.25))
x_max = (-min(trajectory[:, 2])+(range_x*0.25))
y_min = (min(trajectory[:, 0])-(range_y*0.25))
y_max = (max(trajectory[:, 0])+(range_y*0.25))

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

def animate(i):
    x = -trajectory[:i+1, 2]  # Include the current point
    y = trajectory[:i+1, 0]
    line.set_data(x, y)
    point.set_data(x[-1], y[-1])
    return line, point

if calculate_fps:
    # Calculate the desired frame rate
    avg_fps = np.mean(fps_list)
else:
    avg_fps = 30

interval = 1000 / avg_fps  # in milliseconds

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(trajectory), interval=interval, blit=True, repeat=True)

# Save the animation as a video file
Writer = animation.FFMpegWriter(fps=avg_fps)
ani.save('trajectory_animation_' + feature_type + '.mp4', writer=Writer)

plt.show()
