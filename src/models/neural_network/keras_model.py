from keras.models import Sequential
from keras.layers import Dense, Dropout

from numpy.random import seed

import numpy as np

   
#Here's what the typical end-to-end workflow looks like, consisting of:
#Training
#Validation on a holdout set generated from the original training data
#Evaluation on the test data (we didn't do this part in here)

#next goals:
   # try using different lyrs
   # different optimizer
   # different loss

#try to run these codes from scratch without using keras (nice to have)


def build_and_compile_model(input,lyrs=[8], act='linear', opt='Adam', dr=0.0, loss = 'binary_crossentropy'):
             # set random seed for reproducibility
              seed(42)
         
              model = Sequential()

              # create first hidden layer
              model.add(Dense(lyrs[0], input_dim=input.shape[1], activation=act))
    
              for i in range(1,len(lyrs)):
                     model.add(Dense(lyrs[i], activation=act))
              # add dropout, default is none
              model.add(Dropout(dr))
              
              # create output layer
              model.add(Dense(1, activation='sigmoid'))  # output layer
              
              model.compile(loss = loss, optimizer=opt, metrics=['accuracy'])
              return model

def train(x_train,y_train,model,epochs, batch_size, validation_split):
        #verbose = 0 if you don't want to print epoch logs
        #20% of training set is hold out for validation 
        training = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return training

