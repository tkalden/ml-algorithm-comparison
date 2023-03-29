# Import the necessary libraries
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Detail Conversation with chatgpt https://chat.openai.com/chat/3bf91048-2c02-4b1c-a4b5-5c615ae3bc2c


def map_cloth_to_model():
    print(os.getcwd())

    # Load the image model
    image_model = cv2.imread('../../data/model.jpg')

    # Load the cloth image
    cloth_image = cv2.imread('../../data/shirt.jpg')

    # Resize the cloth image to match the size of the image model
    cloth_image_resized = cv2.resize(
        cloth_image, (image_model.shape[1], image_model.shape[0]))

    # Convert the cloth image to grayscale
    cloth_image_gray = cv2.cvtColor(cloth_image_resized, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to the cloth image
    ret, cloth_image_thresh = cv2.threshold(
        cloth_image_gray, 100, 255, cv2.THRESH_BINARY)

    # Invert the binary image
    cloth_image_thresh_inv = cv2.bitwise_not(cloth_image_thresh)

    # Apply the inverted binary image as a mask to the image model
    image_model_masked = cv2.bitwise_and(
        image_model, image_model, mask=cloth_image_thresh_inv)

    # Add the cloth image to the masked image model
    result = cv2.add(image_model_masked, cloth_image_resized)

    # Show the result
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()

 # Load the pre-trained Caffe model for upper body detection

# need to detect ROI of body parts - we will use https://github.com/chuanqi305/MobileNet-SSD
# Community pre-trained models: https://github.com/chuanqi305/MobileNet-SSD

# MobileNet-SSD is a type of computer algorithm called a "detection network"
# which is able to find and identify objects within an image.
# This particular implementation of MobileNet-SSD is designed to
# work with the Caffe machine learning framework and comes
# with a set of pre-trained "weights" that have already been trained on a dataset called VOC0712.
#  This means that the algorithm has already learned how to recognize a
# variety of different objects within images and can be used right away without needing to be trained further.
# The "mAP=0.727" refers to the accuracy of the algorithm, with higher values indicating greater accuracy
#  in detecting objects.

# current we don't have access to model_path = "path/to/model.prototxt"
#weights_path = "path/to/model.caffemodel"


def map_cloth_to_model_body_part():
    absolute_path = os.path.dirname(__file__)
    # Load the image of the model
    model_img = cv2.imread('src/data/model.jpg')

    # Load the cloth image
    cloth_image = cv2.imread('src/data/shirt.jpg')

    # Load Caffe model and prototxt file
    model_path = absolute_path + "/MobileNetSSD_deploy.prototxt"
    weights_path = absolute_path + "/MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(
        model_path, weights_path)

    # Pass model image through the network to get predicted bounding boxes
    blob = cv2.dnn.blobFromImage(model_img, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Find the bounding box that corresponds to the upper body of the model
    for i in range(detections.shape[2]):
        print(i)
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # filter out low-confidence predictions
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # "person" class

                # Get the coordinates of the bounding box for the detected object
                box = detections[0, 0, i, 3:7] * np.array(
                    [model_img.shape[1], model_img.shape[0], model_img.shape[1], model_img.shape[0]])
                box = box.astype(int)
                (startX, startY, endX, endY) = box.astype("int")

                # Get the height and width of the bounding box
                box_width = endX - startX
                box_height = endY - startY

                # Compute the upper body region as a fraction of the bounding box height
                upper_body_y = 40 + startY
                upper_body_h = int(0.6 * box_height)
                upper_body_x = startX
                upper_body_w = box_width

                # filter out lower half of the image
                if upper_body_y + upper_body_h/2 < model_img.shape[0]/2:
                    print("Upper body ROI:", upper_body_x,
                          upper_body_y, upper_body_w, upper_body_h)
                    break

    # Resize sweater image to fit the size of the ROI
    sweater_resized = cv2.resize(cloth_image, (upper_body_w, upper_body_h))

    # Use alpha blending to blend the sweater onto the model image
    alpha = 0.5
    beta = 1 - alpha
    blended = cv2.addWeighted(
        model_img[upper_body_y:upper_body_y + upper_body_h, upper_body_x:upper_body_x + upper_body_w], alpha, sweater_resized, beta, 0)

    # Replace the ROI in the model image with the blended sweater image
    model_img[upper_body_y:upper_body_y + upper_body_h,
              upper_body_x:upper_body_x + upper_body_w] = blended

    # Display the final result
    cv2.imshow("Mapped image", model_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # map_cloth_to_model()
    map_cloth_to_model_body_part()
