#Neural Network

#The output of a simple 2-layer neural network is ŷ = sigma(W2*sigma(W1*x + b1) + b2)
#Sigma is the activation function (we will use the sigmoid activation function here)
#ŷ is the output 
#x is the input layer
#W is the weight 
#b is the biases

##Naturally, the right values for the weights and biases determines the strength of the predictions. 
##the process of fine-tuning the weights and biases from the input data is known as training the Neural Network.

##Each iteration of the training process consists of the following steps:

#1. Calculating the predicted output ŷ, known as feedforward
#2. Updating the weights and biases, known as backpropagation


import numpy as np
import matplotlib as plt

#defining the sigmoid function 
def sigmoid(x):
 return 1/(1 + np.exp(-x))

#defining sigmoid derivative
def sigmoid_derivative(x):
    f = 1/(1 + np.exp(-x))
    return f * (1 - f)
    

#Generating the neural network class 
class NeuralNetwork:
    
    #the initialisation method 
    def __init__(self, x, y):
        #instance variables
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)
        
    #adding a feedforward method to the class
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        
    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    

def NeuralLoop(init_x, init_y, iterations): 
    output = []
    new_network = NeuralNetwork(init_x,init_y)
    for i in range(iterations):
        new_network.feedforward()
        new_network.backprop()
        output.append(new_network.output)
        
    return output, new_network.output
init_x_array = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
init_y_array = np.array([[0],[1],[1],[0]])  
new_loop = NeuralLoop(init_x_array, init_y_array, 1500)
print(new_loop[1])