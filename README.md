# Neural-Network-from-scratch-without-any-libraries

This project demonstrates how to build a simple neural network from scratch without relying on high-level machine learning libraries like TensorFlow or PyTorch. The goal is to provide an educational tool for understanding the inner workings of neural networks, including the forward pass, activation functions, and the backpropagation algorithm for updating weights.

## Features

** Custom Neural Network Architecture: ** Implements a basic neural network with customizable layers, neurons, and activation functions.
### Activation Functions: Includes ReLU and Sigmoid activation functions to introduce non-linearity into the network.
### Loss Calculation: Utilizes the Binary Cross-Entropy loss function for evaluating the performance of the network.
### Backpropagation: Demonstrates the backpropagation algorithm for updating network weights based on the loss gradient.

## Technical Details

### Language: Pure Python
### External Libraries: Only numpy for matrix operations, emphasizing the network's functionality without complex machine learning frameworks.
### Components:
### BinaryCrossEntropy: Loss function class for computing the loss and its gradient.
### ReLU and Sigmoid: Activation function classes for adding non-linearity.
### Neural_layer: Class representing a layer in the neural network, capable of performing forward and backward passes.
### Data: Generates synthetic data to train and evaluate the network, providing a hands-on example of neural network training.

## Working

The neural network is structured into two layers: an input layer and an output layer, with ReLU and Sigmoid activations respectively. The network processes synthetic data, adjusting its weights through backpropagation based on the Binary Cross-Entropy loss.

Initialization: The network initializes with random weights and zero biases.
Forward Pass: Data is passed through the network, with activations applied.
Backward Pass: The network computes gradients and updates weights based on the loss.
Prediction and Evaluation: The network makes predictions on the input data, and the accuracy is calculated.
