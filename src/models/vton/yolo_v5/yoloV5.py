import urllib.request

import cv2
import numpy as np
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# Load the image of the model
img = cv2.imread('src/data/model.jpg')


# Detect objects in the image using YOLOv5
results = model(img)
print(results.pandas().xyxy[0]['name'] == 'person')
# Extract the bounding boxes and confidence scores for all "person" objects

person_results = results.pandas(
).xyxy[0]


# Get the height and width of the image
h, w = img.shape[:2]


# Loop through all the detected person objects
for i, (_, row) in enumerate(person_results.iterrows()):
    # Get the coordinates of the bounding box
    x1, y1, x2, y2 = row[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)

    # Compute the height and width of the bounding box
    bbox_height = y2 - y1
    bbox_width = x2 - x1

    # Compute the coordinates of the upper body region as a fraction of the bounding box height
    upper_body_y = int(y1 + 0.2 * bbox_height)
    upper_body_h = int(0.4 * bbox_height)
    upper_body_x = x1
    upper_body_w = bbox_width

    # Extract the upper body region from the image
    upper_body_roi = img[upper_body_y:upper_body_y +
                         upper_body_h, upper_body_x:upper_body_x + upper_body_w]

    # Load the sweater image
    sweater_img = cv2.imread('src/data/shirt.jpg')

    # Resize the sweater image to match the size of the upper body region
    sweater_img = cv2.resize(sweater_img, (upper_body_w, upper_body_h))

    # Blend the sweater image with the upper body region
    alpha = 0.8
    beta = 1.0 - alpha
    blended_roi = cv2.addWeighted(
        upper_body_roi, alpha, sweater_img, beta, 0.0)

    # Replace the upper body region in the original image with the blended image
    img[upper_body_y:upper_body_y + upper_body_h,
        upper_body_x:upper_body_x + upper_body_w] = blended_roi

# Display the final image
cv2.imshow('Yolo Method', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
