import cv2
import numpy as np

from openpose import pyopenpose as op


class cloth_model:
    def __init__(self):
        self.model = cv2.imread("src/data/model.jpg")
        self.shirt = cv2.imread("src/data/shirt.jpg")
        self.upper_body_keypoints = []

    def start_op_wrapper(self):

        # Initialize OpenPose
        params = {
            # Path to OpenPose models folder
            'model_folder': '/Users/tenzinkalden/projects/mlAlgoComparison/openpose/models',
            'model_pose': 'BODY_25',  # Select pose model
            'net_resolution': '-1x368',  # Input resolution of the network
            'alpha_pose': 0.6,  # Alpha blending value for pose estimation
            'scale_gap': 0.3,  # Scaling gap between scales of the image pyramid
            'scale_number': 1,  # Number of scales to use in the image pyramid
            'part_candidates': True  # Enable part candidates
        }
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        return opWrapper

    def get_segmentation_key_points(self):
        opWrapper = self.start_op_wrapper()
        # Run OpenPose on the image
        datum = op.Datum()
        datum.cvInputData = self.model
        opWrapper.emplaceAndPop([datum])

        # Extract pose keypoints and segmentation map
        keypoints = datum.poseKeypoints
        # segmentation = datum.segmentation getting error Did you forget to `#include <pybind11/stl.h>`? Or <pybind11/complex.h>,
        return keypoints

    def set_upper_body_key_points(self):
        opWrapper = self.start_op_wrapper()

        # Create VectorDatum object
        datum_vec = op.VectorDatum()

        # Create Datum object
        datum = op.Datum()

        # Convert the input image to RGB and pass it to OpenPose for detection
        image_rgb = cv2.cvtColor(self.model, cv2.COLOR_BGR2RGB)
        datum.cvInputData = image_rgb

        # Append Datum to VectorDatum
        datum_vec.append(datum)

        # Process image
        opWrapper.emplaceAndPop(datum_vec)

        # Get pose keypoints
        keypoints = datum.poseKeypoints
        self.upper_body_keypoints = keypoints[:, 1:9, :].astype(np.float32)

    def crop_image(self):
        gray = cv2.cvtColor(self.shirt, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = self.shirt[y:y+h, x:x+w]
        return cropped_image

    def draw_key_points(self):
        # Draw the keypoints on the overlay image
        for i in range(self.upper_body_keypoints.shape[0]):
            for j in range(self.upper_body_keypoints.shape[1]):
                cv2.circle(
                    self.model, (int(self.upper_body_keypoints[i, j, 0]), int(self.upper_body_keypoints[i, j, 1])), 4, (0, 0, 255), -1)

    def mask_target_cloth(self):
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.shirt, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to create a binary image
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Find the contours in the binary image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask of the clothing item
        mask = np.zeros_like(self.shirt)
        cv2.drawContours(mask, contours, 0, (255, 255, 255), -1)
        return mask
