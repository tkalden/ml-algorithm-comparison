import pandas as pd
from ...utility.constant import titanic_folders
import os 

def load_data():
    train = pd.read_csv(titanic_folders["train"])
    test = pd.read_csv(titanic_folders["test"])
    return pd.concat([train, test], axis=0, sort=True)            

def train_test_data(df,output_label):
    x_train = df[pd.notnull(df[output_label])].drop([output_label], axis=1)
    y_train = df[pd.notnull(df[output_label])][[output_label]]
    x_test = df[pd.isnull(df[output_label])].drop([output_label], axis=1)
    return [x_train, y_train,x_test]

def split_training_validation(validation_split,x_train,y_train):
    #split training data into training and validation set
    x_training = x_train.sample(frac = 1 - validation_split)
    x_validation = x_train.drop(x_training.index)
    
    y_training = y_train.sample(frac = 1 - validation_split)
    y_validation = y_train.drop(y_training.index)
    
    return [x_training,y_training,x_validation,y_validation]
