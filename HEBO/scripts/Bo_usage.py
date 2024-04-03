from BO_methods import CustomTask
from config_NN_methods import flex_NN, activation_functions, loss_functions
from BO_pipeline import WeightAndBiasOptimizer
from HEBO.MCBO.mcbo.optimizers.bo_builder import BoBuilder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# Function to preprocess the data
def preprocess_data(df, preds_list, target_var):
    # Drop rows with missing values in specific columns
    df = df.dropna(subset=preds_list)
    target = df[target_var] # store target variable
    # Encode categorical variable 'Sex' into dummy variables
    encode_cat_var = pd.get_dummies(df[preds_list[0]])
    # Drop the original 'Sex' column and concatenate the encoded columns with the DataFrame
    df = df[preds_list]
    df = df.drop([preds_list[0]], axis=1)
    X = pd.concat([encode_cat_var, df], axis=1)
    # Scale the features using Min-Max scaling
    X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
    return X, target


# Load and preprocess data
df = pd.read_csv('train.csv')

X, target_var = preprocess_data(df, ['Sex', 'Age', 'Fare'], 'Survived')
y = target_var
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define neural network layer sizes
layer_sizes = [X.shape[1], 8, 1]

# Neural network objects and activation functions
flex_NN_obj_train = flex_NN(layer_sizes, X_train, y_train)
activation_obj = activation_functions()
parameters = flex_NN_obj_train.label_parameters()
hidden_activation = activation_obj.relu
output_activation = activation_obj.sigmoid
loss_fn_train = loss_functions(y_train).accuracy

# CustomTask instantiation
task = CustomTask(layer_sizes, X_train, y_train)
searchspace = task.get_search_space()

# Build optimizer
optimizer_builder = BoBuilder(model_id='gp_rd', acq_opt_id='is', acq_func_id='ei', tr_id='basic')
opt = optimizer_builder.build_bo(search_space=searchspace, n_init=100)
weight_bias_optimizer = WeightAndBiasOptimizer(task=task, optimizer=opt)

# Optimize weights and biases
weights_and_biases_list, loss_coeff_list = weight_bias_optimizer.weights_biases_loss_coeff(layer_sizes, 50, X_train, y_train, loss_fn_train)
best_weights_and_biases, best_loss_coeff = weight_bias_optimizer.find_best_weights_and_biases(weights_and_biases_list, loss_coeff_list, metric='max')

# Evaluate on test data
flex_NN_obj_test = flex_NN(layer_sizes, X_test, y_test)
ML_model_preds = flex_NN_obj_test.forward_prop(hidden_activation, output_activation, best_weights_and_biases)
loss_fn_test = loss_functions(y_test).accuracy(ML_model_preds)

# Print results
print("Best weights and biases:", best_weights_and_biases)
print("Training set accuracy:", best_loss_coeff)
print("Test set accuracy:", loss_fn_test)
