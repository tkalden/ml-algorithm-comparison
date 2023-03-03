
from src.models import neural_network_keras
from src.models import neural_network
import src.data.process_data as process_data
from src.data.train_test_data import load_data, train_test_data , split_training_validation
from src.utility.graphs import accuracy_plot
import pandas as pd
import numpy as np


def get_data():
    data = load_data()
    process_class = process_data.clean_data(data)
    continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Family_Size']
    process_class.scale_continous_variable(continuous)
    process_class.encode_data(['Sex'])
    process_class.drop_colums(['Cabin', 'Name', 'Ticket', 'PassengerId'])
    process_class.hot_endcode_data(['Embarked', 'Title'])
    processed_data = process_class.return_processed_data()
    return train_test_data(processed_data,'Survived')  

    
if __name__ == "__main__":
    #define constants
    batch_size = 32
    epochs = 100
    validation_split = 0.2
    [x_train, y_train,x_test] = get_data()

    #using the keras library
    print("################# USING TENSORFLOW AND KERAS LIBRARY")
    model = neural_network_keras.build_and_compile_model(x_train)
    training = neural_network_keras.train(x_train,y_train,model,epochs,batch_size,validation_split)
    #accuracy_plot.plot(training)
    
    #using functions built from scratch
    print("################# USING FUNCTIONS BUILT FROM SCRATCH")
    [x_training,y_training,x_validation,y_validation] = split_training_validation(validation_split,x_train,y_train)
    nn = neural_network.optimizer(x_training,y_training,x_validation,y_validation,[8,1],['linear','sigmoid'],2,batch_size,epochs)
    nn.nn_build_model()





