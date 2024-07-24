import cv2
import pyrealsense2 as rs
import numpy as np

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

try:
    while True:
        # Get the frames
        frames = pipeline.wait_for_frames()

        # Get the left infrared frame
        left_ir_frame = frames.get_infrared_frame(1)
        left_ir_image = np.asanyarray(left_ir_frame.get_data())

        # Get the right infrared frame
        right_ir_frame = frames.get_infrared_frame(2)
        right_ir_image = np.asanyarray(right_ir_frame.get_data())

        # Display the frames
        cv2.imshow('Left IR', left_ir_image)
        cv2.imshow('Right IR', right_ir_image)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and release resources
    pipeline.stop()
    cv2.destroyAllWindows()