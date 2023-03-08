import pandas as pd
import numpy as np
from src.models.xception import train, predict

    
if __name__ == "__main__":
    #define constants
    image_size = (299, 299)
    input_shape = (299,299,3)
    batch_size = 32
    epochs = 30
    learning_rate = 0.01
    droprate = 0.3
    hat_path = 'src/data/trouser.jpg'

    #xception to classify clothing types
    train.train_model(image_size,batch_size,epochs,learning_rate,droprate,input_shape)
    print(predict.evaluate_model(image_size))
    print(predict.predict(hat_path,image_size))






