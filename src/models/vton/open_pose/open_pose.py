import os
import sys

import cv2
import numpy as np

from openpose import pyopenpose as op

sys.path.append(
    '../../python')


# Load the input image and the shirt image
input_image = cv2.imread(
    "/Users/tenzinkalden/projects/mlAlgoComparison/src/data/model.jpg")
shirt_image = cv2.imread(
    "/Users/tenzinkalden/projects/mlAlgoComparison/src/data/shirt.jpg", cv2.IMREAD_UNCHANGED)

openpose_path = os.path.dirname(os.path.abspath(op.__file__))
print(f"OpenPose path: {openpose_path}")
model_folder = os.path.join(openpose_path, "models")
print(f"Model folder: {model_folder}")
current_file_path = os.path.abspath(__file__)

# Initialize the OpenPose model parameters
params = dict()
params["model_folder"] = "/Users/tenzinkalden/projects/mlAlgoComparison/openpose/models"
print(current_file_path)
params["face"] = False
params["hand"] = False


# Start the OpenPose model
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Create VectorDatum object
datum_vec = op.VectorDatum()

# Create Datum object
datum = op.Datum()

# Convert the input image to RGB and pass it to OpenPose for detection
image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
datum.cvInputData = image_rgb

# Append Datum to VectorDatum
datum_vec.append(datum)

# Process image
opWrapper.emplaceAndPop(datum_vec)

# Get pose keypoints
keypoints = datum.poseKeypoints

# Get the upper body keypoints by selecting the relevant keypoints
upper_body_keypoints = keypoints[:, 1:9, :]

# Compute the bounding box of the upper body keypoints
x, y, w, h = cv2.boundingRect(np.int32(upper_body_keypoints.reshape(-1, 2)))

# Resize the shirt image to match the dimensions of the input image

overlay_image = input_image.copy()
print(overlay_image.shape)
if w < overlay_image.shape[1]:
    shirt_resized = cv2.resize(shirt_image, (w, h))
else:
    shirt_resized = cv2.resize(shirt_image, (overlay_image.shape[1], h))

# upper body roi

upper_body_roi = overlay_image[y:y+h, x:x+w]0
print(upper_body_roi.shape)

# Add the masked shirt image to the masked input image using alpha blending
alpha = 0.6
output_masked = cv2.addWeighted(
    upper_body_roi, 1.0-alpha, shirt_resized, alpha, 0)
print(output_masked.shape)

# Replace the shirt region in the overlay image with the alpha blended image
overlay_image[y:y+h, x:x+w] = output_masked
# Show the final result
cv2.imshow("Open-Pose-Method", overlay_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
