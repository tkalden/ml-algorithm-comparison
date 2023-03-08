import tensorflow as tf
import numpy as np
from ...utility.constant import clothing_folders, classifier_name, labels


keras = tf.keras
image = tf.keras.preprocessing.image 
xception = tf.keras.applications.xception

# Load the pre-trained model
model = tf.keras.models.load_model(classifier_name)

def predict(path,image_size):
    img = image.load_img(path, target_size=(image_size))
    x = np.array(img)
    X = np.array([x])
    X = xception.preprocess_input(X)
    pred = model.predict(X)
    return labels[pred[0].argmax()]



def evaluate_model(image_size):
    test_gen = image.ImageDataGenerator(preprocessing_function=xception.preprocess_input)
    test_ds = test_gen.flow_from_directory(
        clothing_folders["test"],
        shuffle=False,
        target_size=image_size,
        batch_size=32,
    )

    return model.evaluate(test_ds)

