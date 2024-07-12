import os

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from BO_methods import CustomTask
from config_NN_methods import flex_NN, activation_functions, loss_functions
from BO_pipeline import WeightAndBiasOptimizer
from HEBO.MCBO.mcbo.optimizers.bo_builder import BoBuilder


class cleaning_methods:
    def __init__(self, df):
        """
        Initialize with a DataFrame.
        :param df: pd.DataFrame
        """
        self.df = df

    def remove_outliers(self, lower_lim: float, upper_lim: float) -> 'pd.DataFrame':
        """
        Remove outliers from the DataFrame.
        :param lower_lim: float, lower quantile limit
        :param upper_lim: float, upper quantile limit
        :return: pd.DataFrame, filtered DataFrame without outliers
        """
        Q1 = self.df.quantile(lower_lim)
        Q3 = self.df.quantile(upper_lim)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_filtered = self.df[~((self.df < lower_bound) | (self.df > upper_bound)).any(axis=1)]
        return df_filtered

    def remove_rows(self, col_name: str, wanted_rows_each_categ: int, seed: int = None) -> 'pd.DataFrame':
        """
        Remove rows to balance the DataFrame based on the specified column.
        :param col_name: str, column name to balance
        :param wanted_rows_each_categ: int, number of rows desired for each category
        :param seed: int, random seed for reproducibility
        :return: pd.DataFrame, balanced DataFrame
        """
        if seed is not None:
            np.random.seed(seed)

        values_dict = dict(self.df[col_name].value_counts())
        max_key = max(values_dict, key=values_dict.get)
        min_key = min(values_dict, key=values_dict.get)
        max_val = values_dict[max_key]
        min_val = values_dict[min_key]

        rows_to_remove_from_max = max_val - wanted_rows_each_categ
        rows_to_remove_from_min = min_val - wanted_rows_each_categ

        rows_to_remove_max = []
        rows_to_remove_min = []

        if rows_to_remove_from_max > 0:
            max_indices = self.df[self.df[col_name] == max_key].index.tolist()
            np.random.shuffle(max_indices)
            rows_to_remove_max = max_indices[:rows_to_remove_from_max]

        if rows_to_remove_from_min > 0:
            min_indices = self.df[self.df[col_name] == min_key].index.tolist()
            np.random.shuffle(min_indices)
            rows_to_remove_min = min_indices[:rows_to_remove_from_min]

        rows_to_remove_tot = self.df.loc[rows_to_remove_max + rows_to_remove_min]
        new_df = self.df.drop(rows_to_remove_tot.index)
        new_df.reset_index(drop=True, inplace=True)

        return new_df


class NN_optimizer:
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize with training and testing datasets.
        :param X_train: np.ndarray, training features
        :param X_test: np.ndarray, testing features
        :param y_train: np.ndarray, training labels
        :param y_test: np.ndarray, testing labels
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def build_and_train_nnbp(self, nr_epochs: int, batches: int) -> tuple:
        """
        Build and train a neural network using backpropagation.
        :param nr_epochs: int, number of epochs
        :param batches: int, batch size
        :return: tuple, training and testing accuracy
        """
        model = Sequential([
            Dense(self.X_train.shape[1], activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(self.X_train, self.y_train, epochs=nr_epochs,
                            batch_size=batches, validation_data=(self.X_test, self.y_test), verbose=0)
        loss, accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
        return history.history['accuracy'][-1], accuracy

    def optimize_bo(self, layer_sizes: list, init_par: int, nr_epochs: int) -> tuple:
        """
        Optimize neural network using Bayesian Optimization.
        :param layer_sizes: list, sizes of the neural network layers
        :param init_par: int, number of initial parameters
        :param nr_epochs: int, number of epochs for training
        :return: tuple, best weights and biases, and corresponding accuracy
        """
        task = CustomTask(layer_sizes, self.X_train, self.y_train)
        searchspace = task.get_search_space()
        optimizer_builder = BoBuilder(model_id='gp_rd', acq_opt_id='is', acq_func_id='ei', tr_id='basic')
        opt = optimizer_builder.build_bo(search_space=searchspace, n_init=init_par)
        weight_bias_optimizer = WeightAndBiasOptimizer(task=task, optimizer=opt)
        loss_fn_train = loss_functions(self.y_train).accuracy
        weights_and_biases_list, loss_coeff_list = weight_bias_optimizer.weights_biases_loss_coeff(
            layer_sizes, nr_epochs, self.X_train, self.y_train, loss_fn_train)
        best_weights_and_biases, best_loss_coeff = weight_bias_optimizer.find_best_weights_and_biases(
            weights_and_biases_list, loss_coeff_list, metric='max')
        flex_NN_obj_test = flex_NN(layer_sizes, self.X_test, self.y_test)
        ML_model_preds = flex_NN_obj_test.forward_prop(
            activation_functions.relu, activation_functions.sigmoid, best_weights_and_biases)
        train_acc = best_loss_coeff
        test_acc = loss_functions(self.y_test).accuracy(ML_model_preds)
        return train_acc, test_acc
