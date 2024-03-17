import numpy as np
import pandas as pd
from config_NN_methods import flex_NN, activation_functions


class WeightAndBiasOptimizer:
    def __init__(self, task, optimizer):
        self.task = task
        self.optimizer = optimizer

    def weights_biases_loss_coeff(self, layers_list, epochs,
                                     X_data, y_data, loss_fn):
        weights_and_biases_list = []
        loss_fn_res_list = []
        flex_nn_obj= flex_NN(layers_list, X_data, y_data)
        for it in range(epochs):
            x_next = self.optimizer.suggest()
            x_next_reshaped = np.array(x_next).reshape(1, -1)
            x_next_df = pd.DataFrame(x_next_reshaped, columns=self.task.get_parameter_names())
            #y_next = self.task.evaluate(x_next_df)[1]
            weights_and_biases = self.task.evaluate(x_next_df)
            weights_and_biases_list.append(weights_and_biases)
            output_forward_prop = flex_nn_obj.forward_prop(
                                                          activation_functions.relu,
                                                           activation_functions.sigmoid,
                                                            weights_and_biases)

            loss_fn_res = loss_fn(output_forward_prop) #num_correct / y_data.size
            loss_fn_res_list.append(loss_fn_res)
        return weights_and_biases_list, loss_fn_res_list

    @staticmethod
    def find_best_weights_and_biases(weights_and_biases_list, loss_coeff_list, metric='min'):
        if metric == 'min':
            best_epoch_index = np.argmin(loss_coeff_list)
        elif metric == 'max':
            best_epoch_index = np.argmax(loss_coeff_list)
        else:
            raise ValueError("Invalid metric type. Metric must be 'min' or 'max'.")

        best_weights_and_biases = weights_and_biases_list[best_epoch_index]
        best_loss_coeff = loss_coeff_list[best_epoch_index]

        return best_weights_and_biases, best_loss_coeff



