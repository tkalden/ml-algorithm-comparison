import numpy as np
import tensorflow as tf

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

class optimizer() :
      # init method or constructor
    def __init__(self,x_training,y_training,x_validation,y_validation,lyrs,methods,n_layers,batch_size,epochs):
        self.alpha = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = np.exp(-9)
        #transpose so that each column represent a single sample
        self.x_training = x_training.T 
        self.y_training = y_training.T
        self.x_validation = x_validation.T
        self.y_validation = y_validation.T
        self.shuffle_buffer = 500
        self.layers = [self.x_training.shape[0]] +  lyrs  #first layer is the input features
        self.methods = methods
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.model = None
        self.grads = {}
        self.parameters = {}
        self.exponential_avg = {}
        self.epochs = epochs
        self.caches = {}


    def nn_model_initialize_parameters(self):
        for l in range(1,self.n_layers + 1):
            self.parameters["W"+ str(l)]  =   np.random.randn(self.layers[l],self.layers[l-1])
            self.parameters["b"+ str(l)] = np.zeros((self.layers[l],1),dtype=float)
            self.exponential_avg["Vdw" + str(l)] = np.zeros((self.layers[l],self.layers[l-1]),dtype=float)
            self.exponential_avg["Vdb" + str(l)] = np.zeros((self.layers[l],1),dtype=float)
            self.exponential_avg["Sdw" + str(l)] = np.zeros((self.layers[l],self.layers[l-1]),dtype=float)
            self.exponential_avg["Sdb" + str(l)] = np.zeros((self.layers[l],1),dtype=float)
    
    def nn_model_initialize_gradients(self,n_l_previous, n_l_current,sample_size):
        s_dw = np.zeros(n_l_current,n_l_previous)
        s_db = np.zeros(n_l_current,sample_size)
        return [s_dw,s_db]

    def nn_model_z_function(self,input,weight_matrix, b):
        return np.dot(weight_matrix,input) + b

    def nn_model_activation(self,method,a):
        if method == 'relu':
            return np.maximum(0, a)
        elif method == 'linear':
            return a
        return 1/(1 + np.exp(-a))
        
    def nn_forward_propagation(self,input):
        self.caches["A"+ str(0)] = input
        for l in range(1, self.n_layers + 1): 
          z = self.nn_model_z_function(self.caches["A"+ str(l-1)],self.parameters["W" + str(l)],self.parameters["b" + str(l)])
          self.caches["Z" + str(l)] = z
          self.caches["A"+str(l)] =  self.nn_model_activation(self.methods[l-1],z)
        return self.caches["A" + str(self.n_layers)]
    
    def derivate(self, method, z):
        if method=='relu':
            return 1
        else :
          return 1/(1 + np.exp(-z)) * ( 1 - 1/(1 + np.exp(-z)))
        
    def sum(self, y, a):
        d_al = 0
        for i in range(0,y.shape[0]):
            d_al+= (-y[i]/a[i]) + (1-y[i])/(1-a[i])
        return d_al
    
    def nn_backward_propagation(self,data):
        X = data[:-1]
        Y = data[-1:]
        output = self.nn_forward_propagation(X)
        self.caches["dA" + str(self.n_layers)] = self.sum(Y, output )
        m = Y.shape[1]
        for l in range(self.n_layers,0,-1):
            dZ = self.caches["dA" + str(l)] * self.derivate(self.methods[l-1], self.caches["Z" + str(l)])
            self.grads["dW" +str(l)] = (1/m) * np.dot(dZ , self.caches["A" + str(l-1)].T)
            self.caches["dA"+str(l-1)] = np.dot(self.parameters["W" + str(l)].T ,dZ)
            self.grads["db"+str(l)] = (1/m)*np.sum(dZ,axis =1 , keepdims=True)

    def update_parameters(self,optimization,t):
        for l in range(1,self.n_layers+1):
            if optimization == "GD":
                self.parameters["W"+ str(l)] = self.parameters["W"+ str(l)] - self.alpha * self.grads["dW"+str(l)]
                self.parameters["b"+ str(l)] = self.parameters["b"+ str(l)] - self.alpha * self.grads["db"+str(l)]
            elif optimization == "momentum":
                 self.exponential_avg["Vdw" + str(l)] = self.beta1*self.exponential_avg["Vdw" + str(l)] + (1-self.beta1)*self.grads["dW" + str(l)]
                 self.exponential_avg["Vdb" + str(l)] = self.beta1*self.exponential_avg["Vdb" + str(l)] + (1-self.beta1)*self.grads["db" + str(l)]
                 self.parameters["W"+ str(l)] = self.parameters["W"+ str(l)] - self.alpha * self.exponential_avg["Vdw" + str(l)]
                 self.parameters["b"+ str(l)] = self.parameters["b"+ str(l)] - self.alpha * self.exponential_avg["Vdb" + str(l)]
            elif optimization == "RMSprop":
                 self.exponential_avg["Sdw" + str(l)] = self.beta2*self.exponential_avg["Sdw" + str(l)] + (1-self.beta2)*self.grads["dW" + str(l)]*self.grads["dW" + str(l)]
                 self.exponential_avg["Sdb" + str(l)] = self.beta2*self.exponential_avg["Sdb" + str(l)] + (1-self.beta2)*self.grads["db" + str(l)]*self.grads["db" + str(l)]
                 self.parameters["W"+ str(l)] = self.parameters["W"+ str(l)] - self.alpha * self.grads["dW" + str(l)]/ np.sqrt(self.exponential_avg["Sdw" + str(l)] + self.epsilon)
                 self.parameters["b"+ str(l)] = self.parameters["b"+ str(l)] - self.alpha *  self.grads["db" + str(l)]/ np.sqrt(self.exponential_avg["Sdb" + str(l)] + self.epsilon)
            elif optimization == "Adam":
                 self.exponential_avg["Vdw" + str(l)] = self.beta1*self.exponential_avg["Vdw" + str(l)] + (1-self.beta1)*self.grads["dW" + str(l)]
                 self.exponential_avg["Vdb" + str(l)] = self.beta1*self.exponential_avg["Vdb" + str(l)] + (1-self.beta1)*self.grads["db" + str(l)]
                 self.exponential_avg["Sdw" + str(l)] = self.beta2*self.exponential_avg["Sdw" + str(l)] + (1-self.beta2)*self.grads["dW" + str(l)]*self.grads["dW" + str(l)]
                 self.exponential_avg["Sdb" + str(l)] = self.beta2*self.exponential_avg["Sdb" + str(l)] + (1-self.beta2)*self.grads["db" + str(l)]*self.grads["db" + str(l)]
                 self.exponential_avg["Vdw_corrected" + str(l)] =  self.exponential_avg["Vdw" + str(l)]/(1-self.beta1**t) 
                 self.exponential_avg["Vdb_corrected" + str(l)] =  self.exponential_avg["Vdb" + str(l)]/(1-self.beta1**t)         
                 self.exponential_avg["Sdw_corrected" + str(l)] =  self.exponential_avg["Sdw" + str(l)]/(1-self.beta2**t)         
                 self.exponential_avg["Sdb_corrected" + str(l)] =  self.exponential_avg["Sdb" + str(l)]/(1-self.beta2**t) 
                 self.parameters["W"+ str(l)] = self.parameters["W"+ str(l)] - self.alpha * self.exponential_avg["Vdw_corrected" + str(l)]/np.sqrt(self.exponential_avg["Sdw_corrected" + str(l)] + self.epsilon)
                 self.parameters["b"+ str(l)] = self.parameters["b"+ str(l)] - self.alpha * self.exponential_avg["Vdb_corrected" + str(l)]/np.sqrt(self.exponential_avg["Sdb_corrected" + str(l)] + self.epsilon)
        return 'Done'

        
    def create_mini_batches(self):
      mini_batches = []
      data = np.vstack((self.x_training,self.y_training))
      np.random.shuffle(data)
      i = 0
      for i in range(data.shape[0]+1):
          start = i*self.batch_size
          end = (i+1)*self.batch_size
          mini_batch = data[:,start:end]  
          x_mini = mini_batch[:-1]
          y_mini = mini_batch[-1:]
          mini_batches.append(np.vstack((x_mini,y_mini)))
      if data.shape[0] % self.batch_size !=0 :
           mini_batch = data[:,i*self.batch_size:data.shape[1]]
           x_mini = mini_batch[:-1]
           y_mini = mini_batch[-1:]
           mini_batches.append(np.vstack((x_mini,y_mini)))
      return mini_batches
          
    def nn_build_model(self):
         mini_batches = self.create_mini_batches()
         self.nn_model_initialize_parameters()
         for epoch in range(0, self.epochs):   
            for t in range (0, len(mini_batches)):
                self.nn_backward_propagation(mini_batches[t])
                self.update_parameters("Adam",t+1)   
            y_predicted = self.predict(self.x_training)
            loss = self.binary_cross_entropy(y_predicted,self.y_training)
            accuracy = self.accuracy(y_predicted,self.y_training)
            y_validation_prediction = self.predict(self.x_validation)
            val_loss= self.binary_cross_entropy(y_validation_prediction,self.y_validation)
            val_accuracy = self.accuracy(y_validation_prediction,self.y_validation)
            print("Epoch", epoch+1)
            print("%s: %.2f - %s: %.2f - %s: %.2f - %s: %.2f " % ('loss', loss, 'accuracy',accuracy, 'val_loss',val_loss,'val_accuracy',val_accuracy))


    def predict(self, features):
        self.nn_forward_propagation(features) 
        return self.caches["A"+str(self.n_layers)]
    
    def accuracy(self,y_predicted,y_actual):
       y_predicted = (y_predicted > 0.5) * 1
       total = y_predicted.shape[1]
       return y_actual.eq(y_predicted).T.sum()/total

    def binary_cross_entropy(self,y_pred,y_true): 
        backend = tf.keras.backend
        epsilon = backend.epsilon()
        y_pred = backend.clip(y_pred, epsilon, 1 - epsilon)
        term_0 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        term_1 =  y_true * np.log(y_pred + epsilon)
        return -(term_0 + term_1).T.mean()

        

   
     
