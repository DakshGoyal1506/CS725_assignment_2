# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# # Install numpy
# install("matplotlib")

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Returns the ReLU value of the input x
def relu(x):
    return max(0, x)

# Returns the derivative of the ReLU value of the input x
def relu_derivative(x):
    return (x>0).astype(int)

## TODO 1a: Return the sigmoid value of the input x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    pass

## TODO 1b: Return the derivative of the sigmoid value of the input x
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
    pass

## TODO 1c: Return the derivative of the tanh value of the input x
def tanh(x):
    return np.tanh(x)
    pass

## TODO 1d: Return the derivative of the tanh value of the input x
def tanh_derivative(x):
    return 1 - np.tanh(x)**2
    pass

# Mapping from string to function
str_to_func = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative)
}

# Given a list of activation functions, the following function returns
# the corresponding list of activation functions and their derivatives
def get_activation_functions(activations):  
    activation_funcs, activation_derivatives = [], []
    for activation in activations:
        activation_func, activation_derivative = str_to_func[activation]
        activation_funcs.append(activation_func)
        activation_derivatives.append(activation_derivative)
    return activation_funcs, activation_derivatives

class NN:
    def __init__(self, input_dim, hidden_dims, activations=None):
        '''
        Parameters
        ----------
        input_dim : int
            size of the input layer.
        hidden_dims : LIST<int>
            List of positive integers where each integer corresponds to the number of neurons 
            in the hidden layers. The list excludes the number of neurons in the output layer.
            For this problem, we fix the output layer to have just 1 neuron.
        activations : LIST<string>, optional
            List of strings where each string corresponds to the activation function to be used 
            for all hidden layers. The list excludes the activation function for the output layer.
            For this problem, we fix the output layer to have the sigmoid activation function.
        ----------
        Returns : None
        ----------
        '''
        assert(len(hidden_dims) > 0)
        assert(activations == None or len(hidden_dims) == len(activations))
         
        # If activations is None, we use sigmoid activation for all layers
        if activations == None:
            self.activations = [sigmoid]*(len(hidden_dims)+1)
            self.activation_derivatives = [sigmoid_derivative]*(len(hidden_dims)+1)
        else:
            self.activations, self.activation_derivatives = get_activation_functions(activations + ['sigmoid'])

        ## TODO 2: Initialize weights and biases for all hidden and output layers
        ## Initialization can be done with random normal values, you are free to use
        ## any other initialization technique.
        self.weights = []
        self.biases = []

        for i in range(len(hidden_dims) + 1):
            if(i == 0):
                self.weights.append(np.random.normal(0, 1, (input_dim, hidden_dims[i])))
                self.biases.append(np.random.normal(0, 1, (1, hidden_dims[i])))
            elif i == len(hidden_dims):
                self.weights.append(np.random.normal(0, 1, (hidden_dims[i-1], 1)))
                self.biases.append(np.random.normal(0, 1, (1, 1)))
                pass
            else :
                self.weights.append(np.random.normal(0, 1, (hidden_dims[i-1], hidden_dims[i])))
                self.biases.append(np.random.normal(0, 1, (1, hidden_dims[i])))

    def forward(self, X):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        ----------
        Returns : output probabilities, numpy array of shape (N, 1) 
        ----------
        '''
        # Forward pass

        ## TODO 3a: Compute activations for all the nodes with the corresponding
        ## activation function of each layer applied to the hidden nodes

        output_probs = X
        for i in range(len(hidden_dims) + 1):
            output_probs = np.dot(output_probs, self.weights[i]) + self.biases[i]
            output_probs = self.activations[i](output_probs)

        ## TODO 3b: Calculate the output probabilities of shape (N, 1) where N is number of examples


        return output_probs

    def backward(self, X, y):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        y : target labels, numpy array of shape (N, 1) where N is the number of examples
        ----------
        Returns : gradients of weights and biases
        ----------
        '''
        # Backpropagation

        ## TODO 4a: Compute gradients for the output layer after computing derivative of 
        ## sigmoid-based binary cross-entropy loss
        ## Hint: When computing the derivative of the cross-entropy loss, don't forget to 
        ## divide the gradients by N (number of examples)  

        N = X.shape[0]
        output_probs = self.forward(X)
        y = y.reshape(-1, 1)
        d_output = (output_probs - y) / N
        
        
        ## TODO 4b: Next, compute gradients for all weights and biases for all layers
        ## Hint: Start from the output layer and move backwards to the first hidden layer
        self.grad_weights = []
        self.grad_biases = []

        values = [X]
        z = []
        for i in range(len(self.weights)):
            z0 = np.dot(values[-1], self.weights[i]) + self.biases[i]
            z.append(z0)
            values.append(self.activations[i](z0))
        
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(values[i].T, d_output)
            grad_b = np.sum(d_output, axis=0, keepdims=True)
            self.grad_weights.append(grad_w)
            self.grad_biases.append(grad_b)
            if i > 0:
                d_output = np.dot(d_output, self.weights[i].T) * self.activation_derivatives[i-1](z[i-1]) 


        return self.grad_weights[::-1], self.grad_biases[::-1]

    def step_bgd(self, weights, biases, delta_weights, delta_biases, optimizer_params, epoch):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                gd_flag: 1 for Vanilla SGD, 2 for SGD with Exponential, 3 for Momentum
                momentum: Momentum coefficient, used when gd_flag is 3.
                decay_constant: Decay constant for exponential learning rate decay, used when gd_flag is 2.
            epoch: Current epoch number
        '''
        gd_flag = optimizer_params['gd_flag']
        learning_rate = optimizer_params['learning_rate']
        momentum = optimizer_params['momentum']
        decay_constant = optimizer_params['decay_constant']

        updated_W = []
        updated_B = []
        ### Calculate updated weights using methods as indicated by gd_flag

        ## TODO 5a: Variant 1(gd_flag = 1): Vanilla SGD with Static Learning Rate
        ## Use the hyperparameter learning_rate as the static learning rate
        if gd_flag == 1:
            updated_W = [w - learning_rate * dw for w, dw in zip(weights, delta_weights)]
            updated_B = [b - learning_rate * db for b, db in zip(biases, delta_biases)]

        ## TODO 5b: Variant 2(gd_flag = 2): Vanilla SGD with Exponential Learning Rate Decay
        ## Use the hyperparameter learning_rate as the initial learning rate
        ## Use the parameter epoch for t
        ## Use the hyperparameter decay_constant as the decay constant

        # elif gd_flag == 2:
        #     updated_W = [w - learning_rate * np.exp(-decay_constant * epoch) * dw for w, dw in zip(weights, delta_weights)]
        #     updated_B = [b - learning_rate * np.exp(-decay_constant * epoch) * db for b, db in zip(biases, delta_biases)]
        elif gd_flag == 2:
            decayed_learning_rate = learning_rate * np.exp(-decay_constant * epoch)
            
            updated_W = [w - decayed_learning_rate * dw for w, dw in zip(weights, delta_weights)]
            updated_B = [b - decayed_learning_rate * db for b, db in zip(biases, delta_biases)]


        ## TODO 5c: Variant 3(gd_flag = 3): SGD with Momentum
        ## Use the hyperparameters learning_rate and momentum

        elif gd_flag == 3:
            if not hasattr(self, 'momentum_W'):
                self.momentum_W = [np.zeros_like(w) for w in weights]
                self.momentum_B = [np.zeros_like(b) for b in biases]

            self.momentum_W = [momentum * mw + (1 - momentum) * dw for mw, dw in zip(self.momentum_W, delta_weights)]
            self.momentum_B = [momentum * mb + (1 - momentum) * db for mb, db in zip(self.momentum_B, delta_biases)]

            updated_W = [w - learning_rate * mw for w, mw in zip(weights, self.momentum_W)]
            updated_B = [b - learning_rate * mb for b, mb in zip(biases, self.momentum_B)]


        return updated_W, updated_B

    def step_adam(self, weights, biases, delta_weights, delta_biases, optimizer_params):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                beta: Exponential decay rate for the first moment estimates.
                gamma: Exponential decay rate for the second moment estimates.
                eps: A small constant for numerical stability.
        '''
        learning_rate = optimizer_params['learning_rate']
        beta = optimizer_params['beta1']
        gamma = optimizer_params['beta2']
        eps = optimizer_params['eps']       

        ## TODO 6: Return updated weights and biases for the hidden layer based on the update rules for Adam Optimizer

        updated_W = []
        updated_B = []
        
        if not hasattr(self, 'v_W'):
            self.v_W = [np.zeros_like(w) for w in weights]
            self.v_B = [np.zeros_like(b) for b in biases]
            self.s_W = [np.zeros_like(w) for w in weights]
            self.s_B = [np.zeros_like(b) for b in biases]
            self.t = 0

        self.t += 1

        for i in range(len(weights)):
            self.v_W[i] = beta * self.v_W[i] + (1 - beta) * delta_weights[i]
            self.v_B[i] = beta * self.v_B[i] + (1 - beta) * delta_biases[i]

            self.s_W[i] = gamma * self.s_W[i] + (1 - gamma) * (delta_weights[i] ** 2)
            self.s_B[i] = gamma * self.s_B[i] + (1 - gamma) * (delta_biases[i] ** 2)

            s_W_hat = self.s_W[i] / (1 - gamma ** self.t)
            s_B_hat = self.s_B[i] / (1 - gamma ** self.t)

            v_W_hat = self.v_W[i] / (1 - beta ** self.t)
            v_B_hat = self.v_B[i] / (1 - beta ** self.t)

            updated_W.append(weights[i] - (learning_rate * v_W_hat) / (np.sqrt(s_W_hat) + eps))
            updated_B.append(biases[i] - (learning_rate * v_B_hat) / (np.sqrt(s_B_hat) + eps))

        return updated_W, updated_B

    def train(self, X_train, y_train, X_eval, y_eval, num_epochs, batch_size, optimizer, optimizer_params):
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            # Divide X,y into batches
            X_batches = np.array_split(X_train, X_train.shape[0]//batch_size)
            y_batches = np.array_split(y_train, y_train.shape[0]//batch_size)
            for X, y in zip(X_batches, y_batches):
                # Forward pass
                self.forward(X)
                # Backpropagation and gradient descent weight updates
                dW, db = self.backward(X, y)
                if optimizer == "adam":
                    self.weights, self.biases = self.step_adam(
                        self.weights, self.biases, dW, db, optimizer_params)
                elif optimizer == "bgd":
                    self.weights, self.biases = self.step_bgd(
                        self.weights, self.biases, dW, db, optimizer_params, epoch)

            # Compute the training accuracy and training loss
            train_preds = self.forward(X_train)
            train_loss = np.mean(-y_train*np.log(train_preds) - (1-y_train)*np.log(1-train_preds))
            train_accuracy = np.mean((train_preds > 0.5).reshape(-1,) == y_train)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            train_losses.append(train_loss)

            # Compute the test accuracy and test loss
            test_preds = self.forward(X_eval)
            test_loss = np.mean(-y_eval*np.log(test_preds) - (1-y_eval)*np.log(1-test_preds))
            test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
            print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            test_losses.append(test_loss)

        return train_losses, test_losses

    
    # Plot the loss curve
    def plot_loss(self, train_losses, test_losses):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('trial.png')
 

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "data_train.csv"
    eval_file_path = "data_eval.csv"
    
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)

    # Separate the data into X (features) and y (target) arrays
    X_train = data[:, :-1]
    y_train = data[:, -1]
    X_eval = data_eval[:, :-1]
    y_eval = data_eval[:, -1]

    # Create and train the neural network
    input_dim = X_train.shape[1]
    X_train = X_train**2
    X_eval = X_eval**2
    hidden_dims = [4,2, 2] # the last layer has just 1 neuron for classification
    num_epochs = 30
    batch_size = 100
    activations = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
    # optimizer = "bgd"
    # optimizer_params = {
    #     'learning_rate': 0.05,
    #     'gd_flag': 3,
    #     'momentum': 0.9,
    #     'decay_constant': 0.02 # decay constant will also change
    # }
    
    # For Adam optimizer you can use the following
    optimizer = "adam"
    optimizer_params = {
        'learning_rate': 0.01,
        'beta1' : 0.8,
        'beta2' : 0.999,
        'eps' : 1e-8
    }

     
    model = NN(input_dim, hidden_dims)
    train_losses, test_losses = model.train(X_train, y_train, X_eval, y_eval,
                                    num_epochs, batch_size, optimizer, optimizer_params) #trained on concentric circle data 
    test_preds = model.forward(X_eval)

    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Final Test accuracy: {test_accuracy:.4f}")

    model.plot_loss(train_losses, test_losses)
