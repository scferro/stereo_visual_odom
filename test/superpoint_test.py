import cv2
import numpy as np
import torch
from superpoint import SuperPointFrontend

# Initialize the SuperPointFrontend with the appropriate parameters
weights_path = '/home/scferro/Documents/msai495/cv_final_project/SuperPointPretrainedNetwork/superpoint_v1.pth'
nms_dist = 1
conf_thresh = 0.1
nn_thresh = 1.0
cuda = torch.cuda.is_available()

sp = SuperPointFrontend(weights_path, nms_dist, conf_thresh, nn_thresh, cuda=cuda)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and normalize
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = gray_frame.astype(np.float32) / 255.

    # Run SuperPoint model
    pts, desc, heatmap = sp.run(gray_frame)

    # Draw keypoints on the frame
    for i in range(pts.shape[1]):
        pt = (int(pts[0, i]), int(pts[1, i]))
        cv2.circle(frame, pt, 1, (0, 255, 0), -1)

    # Show the frame with keypoints
    cv2.imshow('SuperPoint Keypoints', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
