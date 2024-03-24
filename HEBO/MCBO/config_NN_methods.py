import numpy as np


class flex_NN:
    def __init__(self, layers, X_data, y_data, learning_rate=0.1):
        self.layers = layers
        # self.weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        # 0self.biases = [np.random.randn(1, layers[i+1]) for i in range(len(layers)-1)]
        self.learning_rate = learning_rate
        self.X_data = X_data
        self.y_data = y_data


    def label_parameters(self):
        labels = {}
        for i in range(len(self.layers) - 1):
            labels[f"W{i}"] = f"{i}"
            labels[f"b{i}"] = f"{i}"
        return labels.keys()

    @staticmethod
    def parameters_dict(params_array):
        """
        Create a dictionary of parameters from an array containing weights and biases.

        Parameters:
        params_array -- array containing weights and biases in the order [W0, b0, W1, b1, ..., Wn, bn]

        Returns:
        parameters -- dictionary containing the parameters 'W0', 'b0', 'W1', 'b1', ..., 'Wn', 'bn'
        """
        num_layers = len(params_array) // 2
        parameters = {}
        for i in range(num_layers):
            parameters[f'W{i}'] = params_array[2 * i]
            parameters[f'b{i}'] = params_array[2 * i + 1]
        return parameters

    def blocks(self):
        blocks_info = []
        start_idx = 0
        for i in range(len(self.layers) - 1):
            # Weight block
            weight_block_size = self.layers[i] * self.layers[i + 1]
            weight_block = (start_idx, start_idx + weight_block_size)
            reshape_weight = (self.layers[i + 1], self.layers[i])

            # Bias block
            bias_block = (weight_block[1], weight_block[1] + self.layers[i + 1])
            reshape_bias = (self.layers[i + 1], 1)

            blocks_info.extend([weight_block, reshape_weight, bias_block, reshape_bias])

            start_idx = bias_block[1]  # Update start index for next iteration

        slicing, reshaping = blocks_info[::2], blocks_info[1::2]
        # print(slicing, reshaping, 'slicing reshaping')
        return slicing, reshaping

    def forward_prop(self,  activation_hidden, activation_output, parameters):
        """
        Perform forward propagation through each layer of a neural network.

        Parameters:
        parameters -- dictionary containing the parameters 'W0', 'b0', 'W1', 'b1', ..., 'Wn', 'bn'
        X -- input data (numpy array) of shape (input_size, m)

        Returns:
        A -- output of the last layer
        caches -- list of tuples containing the linear and activation values for each layer
        """
        caches = []
        A = self.X_data.T

        num_layers = len(parameters) // 2
        for i in range(num_layers):
            W = parameters[f'W{i}']
            b = parameters[f'b{i}']
            Z = np.dot(W, A) + b
            #relu_Z, sigmoid_Z = np.maximum(0, Z), 1 / (1 + np.exp(-Z))
            #A = relu_Z if i < num_layers - 1 else sigmoid_Z
            A = activation_hidden(Z) if i < num_layers else activation_output(Z)
            caches.append((Z, A))

        return A.T

class loss_functions:
    def __init__(self, y_data):
        self.y_data = y_data

    def binary_cross_entropy(self, y_pred):
        # Calculate binary cross-entropy loss
        cost = np.mean(-(self.y_data * np.log(y_pred) + (1 - self.y_data) * np.log(1 - y_pred)))
        return cost

    def accuracy(self, y_pred, thresh=0.5):
        # Calculate accuracy
        binary_output = np.where(y_pred < thresh, 0, 1).flatten()
        num_correct = np.sum(binary_output == self.y_data)
        accuracy = num_correct / len(self.y_data)
        return accuracy

    def mean_squared_error(self, y_pred):
        cost = np.mean((np.array(self.y_data) - y_pred) ** 2)  # Mean squared error
        return cost

    def r_squared(self, y_pred):
        ss_res = np.sum((np.array(self.y_data) - y_pred) ** 2)  # Sum of squared residuals
        ss_tot = np.sum((np.array(self.y_data) - np.mean(self.y_data)) ** 2)  # Total sum of squares
        r2 = 1 - (ss_res / ss_tot)  # R-squared
        return r2


class activation_functions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=0))  # Subtract max to avoid overflow
        return exp_x / np.sum(exp_x, axis=0)

    @staticmethod
    def linear(x):
        return x

