import numpy as np


class flex_NN:
    def __init__(self, layers: list, X_data: np.ndarray, y_data: np.ndarray, learning_rate: float = 0.1):
        self.layers = layers
        self.learning_rate = learning_rate
        self.X_data = X_data
        self.y_data = y_data

    def label_parameters(self) -> list:
        labels = {}
        for i in range(len(self.layers) - 1):
            labels[f"W{i}"] = f"{i}"
            labels[f"b{i}"] = f"{i}"
        return list(labels.keys())

    @staticmethod
    def parameters_dict(params_array: np.ndarray) -> dict:
        num_layers = len(params_array) // 2
        parameters = {}
        for i in range(num_layers):
            parameters[f'W{i}'] = params_array[2 * i]
            parameters[f'b{i}'] = params_array[2 * i + 1]
        return parameters

    def blocks(self) -> tuple:
        blocks_info = []
        start_idx = 0
        for i in range(len(self.layers) - 1):
            weight_block_size = self.layers[i] * self.layers[i + 1]
            weight_block = (start_idx, start_idx + weight_block_size)
            reshape_weight = (self.layers[i + 1], self.layers[i])
            bias_block = (weight_block[1], weight_block[1] + self.layers[i + 1])
            reshape_bias = (self.layers[i + 1], 1)
            blocks_info.extend([weight_block, reshape_weight, bias_block, reshape_bias])
            start_idx = bias_block[1]
        slicing, reshaping = blocks_info[::2], blocks_info[1::2]
        return slicing, reshaping

    def forward_prop(self, activation_hidden, activation_output, parameters: dict) -> np.ndarray:
        caches = []
        A = self.X_data.T

        num_layers = len(parameters) // 2
        for i in range(num_layers):
            W = parameters[f'W{i}']
            b = parameters[f'b{i}']
            Z = np.dot(W, A) + b
            A = activation_hidden(Z) if i < num_layers else activation_output(Z)
            caches.append((Z, A))
        return A.T


class loss_functions:
    def __init__(self, y_data: np.ndarray):
        self.y_data = y_data

    def binary_cross_entropy(self, y_pred: np.ndarray) -> float:
        cost = np.mean(-(self.y_data * np.log(y_pred) + (1 - self.y_data) * np.log(1 - y_pred)))
        return cost

    def accuracy(self, y_pred: np.ndarray, thresh: float = 0.5) -> float:
        binary_output = np.where(y_pred < thresh, 0, 1).flatten()
        num_correct = np.sum(binary_output == self.y_data)
        accuracy = num_correct / len(self.y_data)
        return accuracy

    def mean_squared_error(self, y_pred: np.ndarray) -> float:
        cost = np.mean((np.array(self.y_data) - y_pred) ** 2)
        return cost

    def r_squared(self, y_pred: np.ndarray) -> float:
        ss_res = np.sum((np.array(self.y_data) - y_pred) ** 2)
        ss_tot = np.sum((np.array(self.y_data) - np.mean(self.y_data)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


class activation_functions:
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=0))
        return exp_x / np.sum(exp_x, axis=0)

    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return x
