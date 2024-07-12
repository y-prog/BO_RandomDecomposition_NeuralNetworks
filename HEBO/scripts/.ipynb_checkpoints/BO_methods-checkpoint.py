from config_NN_methods import flex_NN
#from MCBO.config_NN_methods import flex_NN
import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict
#from HEBO.MCBO.mcbo.tasks.task_base import TaskBase
#from mcbo.tasks.task_base import TaskBase
from HEBO.MCBO.mcbo.tasks.task_base import TaskBase

class CustomTask(TaskBase):
    def __init__(self, layer_sizes, X_data, y_data):  # pars
        self.layer_sizes = layer_sizes
        self.X_data = X_data
        self.y_data = y_data

    @property
    def name(self) -> str:
        return 'Custom Task'

    def evaluate(self, x: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        flex_nn_obj = flex_NN(self.layer_sizes, self.X_data, self.y_data)
        slices, reshape_dims = flex_nn_obj.blocks()[0], flex_nn_obj.blocks()[1]
        y = np.zeros((len(x), 1))
        for ind in range(len(x)):
            x_ind = x.iloc[ind].to_numpy()  # Convert Series to NumPy array
            sliced_arrays = [x_ind[start:end] for start, end in slices]
            reshaped_arrays = tuple([sliced_array.reshape(shape)
                                     for sliced_array, shape in zip(sliced_arrays, reshape_dims)])
            #feed_forward_output =  flex_nn_obj.forward_prop_sigmoid(flex_nn_obj.parameters_dict(reshaped_arrays))
            #y[ind] = flex_nn_obj.binary_cross_entropy(feed_forward_output)
        return flex_nn_obj.parameters_dict(reshaped_arrays)

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        params = []
        start_idx = 0
        for layer, size in enumerate(self.layer_sizes[:-1]):
            for i in range(size):
                for j in range(self.layer_sizes[layer + 1]):
                    params.append({'name': f'W{layer}_{i}{j}', 'type': 'num', 'lb': -1, 'ub': 1})
            for i in range(self.layer_sizes[layer + 1]):
                params.append({'name': f'b{layer}_{i}', 'type': 'num', 'lb': -1, 'ub': 1})
        return params

    def get_parameter_names(self) -> List[str]:
        params = []
        for layer, size in enumerate(self.layer_sizes[:-1]):
            for i in range(size):
                for j in range(self.layer_sizes[layer + 1]):
                    params.append(f'W{layer}_{i}{j}')
            for i in range(self.layer_sizes[layer + 1]):
                params.append(f'b{layer}_{i}')
        return params