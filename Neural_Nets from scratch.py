import numpy as np

np.random.seed(1)

# Creating fake data
x = np.random.random((5,3)) # 5 batches/samples for 3 different features 
y = np.random.randint(0,2,(5,1)) # Labels with either 0 or 1 for 5 different samples

class BinaryCrossEntropy:
    def forward(self, ytrue, ypred):
        # Handling numerical instability, because i generated a lot of random data 
        ypred = np.clip(ypred, 1e-15, 1 - 1e-15)
        loss = -(ytrue * np.log(ypred) + (1 - ytrue) * np.log(1 - ypred))
        loss_gradient = (ypred - ytrue) / (ypred * (1 - ypred))
        return loss, loss_gradient
    
class ReLU:
    def forward(self,inputs):
        return np.maximum(0,inputs)
    
class Sigmoid:
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

class Neural_layer:
    def __init__(self,num_of_features,num_of_neurons):
        self.weights =  np.random.randn(num_of_features,num_of_neurons)
        self.bias = np.zeros((1,num_of_neurons)) 
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.bias
        return self.output
    def backward(self, inputs, ytrue, ypred, BinaryCrossEntropy , learning_rate = 0.01):
        loss, loss_grad = BinaryCrossEntropy.forward(ytrue, ypred)
        delta_weights = np.dot(inputs.T, loss_grad)
        self.weights -= learning_rate * delta_weights

learning_rate = 0.01
epochs = 11

#Assigning the layers - We will have 2 layers

# First layer
layer1 = Neural_layer(num_of_features=3, num_of_neurons=5)
activation_function = ReLU()
loss_function = BinaryCrossEntropy()

# Second layer
layer2 = Neural_layer(num_of_features=5, num_of_neurons=1)
activation_function2 = Sigmoid()

# Backpropagation and Training Loop
for epoch in range(epochs):
    # For Layer 1   
    layer1_output = layer1.forward(x)
    activation_output = activation_function.forward(layer1_output) 
    layer1.backward(x, y, activation_output, loss_function, learning_rate=learning_rate)

    # For Layer 2
    layer2_output = layer2.forward(activation_output)
    activation_output2 = activation_function2.forward(layer2_output)
    layer2.backward(activation_output, y, activation_output2, loss_function, learning_rate=learning_rate)

# Output values will be the results from the activation function from last layer
outputs = activation_output2

# Thresholding the output values as either 0 or 1
for index,i in enumerate(outputs):
    if i>0.5:
        outputs[index] = 1
    else:
        outputs[index] = 0

accuracy = np.sum(outputs == y) / len(y)

print("Predicted labels:",np.array(outputs).reshape(1,-1))
print("True labels:", y.reshape(1,-1))
print('Accuracy of model:',accuracy*100,'%')
