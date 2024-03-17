from BO_methods import CustomTask  # Import CustomTask from BO_methods module
from config_NN_methods import flex_NN, activation_functions, loss_functions
from BO_pipeline import WeightAndBiasOptimizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from mcbo.optimizers.bo_builder import BoBuilder
import numpy as np


# Define the Eggholder function
def eggholder(x, y):
    return (-(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47)))))

# Define the range of x and y values
x_min, x_max = -512, 512
y_min, y_max = -512, 512

# Define the number of points per dimension
num_points_per_dimension =25

# Generate X1, X2, X3, and X4 coordinates with the specified number of points
x = np.linspace(x_min, x_max, num_points_per_dimension)
y = np.linspace(y_min, y_max, num_points_per_dimension)

# Generate the meshgrid
X, Y = np.meshgrid(x, y)

# Calculate Z values using the Eggholder function
Z_eggholder = eggholder(X, Y)

# Flatten X, Y, and Z to create the predictors
X1_flat = X.flatten()
X2_flat = Y.flatten()
X3_flat = (X ** 2).flatten()
X4_flat = (Y ** 2).flatten()

# Combine X1, X2, X3, and X4 to create the predictors
X = np.column_stack((X1_flat, X2_flat, X3_flat, X4_flat))

# Binarize the Z_eggholder values to create the binary target variable
threshold = np.mean(Z_eggholder)
y = (Z_eggholder.flatten() > threshold).astype(int)

# Display the shapes of predictors and target
print("Shape of predictors:", X.shape)
print("Shape of target:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Usage:
layer_sizes = [4,5,3,1]   # Define your layer sizes

flex_NN_obj_train = flex_NN(layer_sizes, X_train, y_train)
activation_obj = activation_functions()

parameters = flex_NN_obj_train.label_parameters()

hidden_activation = activation_obj.relu
output_activation = activation_obj.sigmoid
loss_fn_train= loss_functions(y_train).accuracy

task = CustomTask(layer_sizes, X_train, y_train)  # Instantiate CustomTask from BO_methods module
searchspace = task.get_search_space()

optimizer_builder = BoBuilder(model_id='gp_rd', acq_opt_id='is', acq_func_id='ei', tr_id='basic')
opt = optimizer_builder.build_bo(search_space=searchspace, n_init=300)
print('opt ========', opt)

weight_bias_optimizer = WeightAndBiasOptimizer(task=task, optimizer=opt)

all_weights_and_biases_and_loss_coeff= weight_bias_optimizer.weights_biases_loss_coeff(
                                layer_sizes, 250, X_train, y_train, loss_fn_train)

# Call the weights_biases_loss_coeff function to generate weights_and_biases_list and loss_coeff_list
weights_and_biases_list, loss_coeff_list =\
    weight_bias_optimizer.weights_biases_loss_coeff(layer_sizes, 50,
                                                    X_train, y_train, loss_fn_train)

# Call the find_best_weights_and_biases function with the generated parameters
best_weights_and_biases, best_loss_coeff = (
    weight_bias_optimizer.find_best_weights_and_biases(weights_and_biases_list,
                                                       loss_coeff_list, metric='max'))

print(best_weights_and_biases, best_loss_coeff)

flex_NN_obj_test=flex_NN(layer_sizes, X_test, y_test)
ML_model_preds = flex_NN_obj_test.forward_prop(hidden_activation, output_activation, best_weights_and_biases)


loss_fn_test = loss_functions(y_test).accuracy(ML_model_preds)
print(loss_fn_test)

