import tensorflow as tf
from .model import xception_model
from ...utility.constant import clothing_folders,classifier_name


keras = tf.keras
xception = tf.keras.applications.xception
image = tf.keras.preprocessing.image 


def train_model(image_size,batch_size,epochs,learning_rate, droprate, input_shape):

    train_gen = image.ImageDataGenerator(
        preprocessing_function=xception.preprocess_input,
        shear_range=10.0,
        zoom_range=0.1,
        horizontal_flip=True,  
    )

    train_ds = train_gen.flow_from_directory(
        clothing_folders["train"],
        seed=1,
        target_size=image_size,
        batch_size=batch_size,
    )

    validation_gen = image.ImageDataGenerator(preprocessing_function=xception.preprocess_input)

    val_ds = validation_gen.flow_from_directory(
        clothing_folders["validation"],
        seed=1,
        target_size=image_size,
        batch_size=batch_size,
    )

    model = xception_model(learning_rate=learning_rate, droprate=droprate,input_shape=input_shape)
 
    model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    model.save(classifier_name)

