
from src.models.neural_network import keras_model,model_from_scratch
import models.neural_network.process_data as process_data
from models.neural_network.train_test_data import load_data, train_test_data , split_training_validation


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
    epochs = 30
    validation_split = 0.2

    [x_train, y_train,x_test] = get_data()

    #using the keras library
    print("################# USING TENSORFLOW AND KERAS LIBRARY")
    model = keras_model.build_and_compile_model(x_train)
    training = keras_model.train(x_train,y_train,model,epochs,batch_size,validation_split)
    #accuracy_plot.plot(training)
    
    #using functions built from scratch
    print("################# USING FUNCTIONS BUILT FROM SCRATCH")
    [x_training,y_training,x_validation,y_validation] = split_training_validation(validation_split,x_train,y_train)
    nn = model_from_scratch.optimizer(x_training,y_training,x_validation,y_validation,[8,1],['linear','sigmoid'],2,batch_size,epochs)
    nn.nn_build_model()







