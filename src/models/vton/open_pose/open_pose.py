import os
import sys

import cv2
import numpy as np
import open_pose_functions as op

op = op.cloth_model()

# Get the upper body keypoints by selecting the relevant keypoints
person_img_h, person_img_w = op.model.shape[:2]

op.set_upper_body_key_points()

upper_body_keypoints = op.upper_body_keypoints
print(upper_body_keypoints)

# Extract the left and right shoulder keypoints
neck = upper_body_keypoints[0][0]
left_shoulder = upper_body_keypoints[0][1]
left_wrist = upper_body_keypoints[0][3]
right_shoulder = upper_body_keypoints[0][4]

shirt_width = int(np.linalg.norm(right_shoulder - left_shoulder))
shirt_height = int(np.linalg.norm(left_shoulder - left_wrist))
print(shirt_width, shirt_height)
shirt = op.crop_image()
shirt_img_resized = cv2.resize(shirt, (shirt_width, shirt_height))

# Rotate shirt image to align with shoulders
angle = np.arctan2(right_shoulder[1] - left_shoulder[1],
                   right_shoulder[0] - left_shoulder[0]) * 180 / np.pi
M = cv2.getRotationMatrix2D((right_shoulder[1], right_shoulder[1]), angle, 1)
shirt_img_rotated = cv2.warpAffine(
    shirt_img_resized, M, (shirt_width, shirt_height))
print(shirt_img_rotated.shape)

# Create a binary mask of the shirt
channels = cv2.split(shirt_img_rotated)
shirt_img_alpha = channels[-1]
_, shirt_mask = cv2.threshold(shirt_img_alpha, 0, 255, cv2.THRESH_BINARY)
shirt_mask_inv = cv2.bitwise_not(shirt_mask)

# Crop the shirt image
shirt_img_cropped = cv2.bitwise_and(
    shirt_img_rotated, shirt_img_rotated, mask=shirt_mask)

# Use alpha blending to blend the sweater onto the model image
upper_body_x = int(left_shoulder[0])
upper_body_y = int(left_shoulder[1])
alpha = 0.5
beta = 1 - alpha
blended = cv2.addWeighted(
    op.model[upper_body_y:upper_body_y + shirt_height, upper_body_x:upper_body_x + shirt_width], alpha, shirt_img_cropped, beta, 0)

# Replace the ROI in the model image with the blended sweater image
op.model[upper_body_y:upper_body_y + shirt_height,
         upper_body_x:upper_body_x + shirt_width] = blended

op.draw_key_points()

# Display the result
#cv2.imshow('Person with Shirt', shirt_img_resized)
cv2.imshow('Person with Shirt', op.model)
cv2.waitKey(0)

cv2.destroyAllWindows()
