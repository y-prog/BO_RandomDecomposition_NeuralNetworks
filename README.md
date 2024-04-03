# Classification predictions with Bayesian Optimization for Multi-layer Perceptrons 
## Description

The goal of this project is Apply a Bayesian Optimization for Multi-layer Perceptrons architecture
to a Feed Forward Neural Network as an alterative to the traditional back-propagation


## Project Structure

- `BO_RandomDecomposition_NeuralNetworks
/HEBO/scripts`: Directory containing Bo_usage.py taking the titanic dataset as input.
  - ``BO_RandomDecomposition_NeuralNetworks
/HEBO/scripts: same folder contains
           - config_NN_methods.py
           - BO_methods.py
           - BO_pipeline.py
    which are utilized in script.py
## Preprocessing Data

This project includes a preprocessing function designed specifically for the Titanic dataset from Kaggle. 
The function handles missing data and encodes categorical variables, such as the 'Sex' column.

**Note:** If you are using the Titanic dataset or a similar dataset with the same structure, you can directly utilize this function for preprocessing
but if the features of your dataset vary, you may need to customize or develop additional preprocessing methods to suit the characteristics and requirements of your data.
You can refer to the preprocessing function as a reference or starting point for building your own preprocessing pipeline.

## Usage

### Setting up the Environment

1. Install required packages, create an activate the conda environment called mcbo_env by running the following commands:
   
   ```
   conda env create -f environment.yml
   conda activate mcbo_env
  


## Getting Started
1. Clone the repository:
   
   git clone https://github.com/y-prog/BO_RandomDecomposition_NeuralNetworks.git

2. open scripts.py in the following directory BO_RandomDecomposition_NeuralNetworks
/HEBO/scripts

3. configure scripts.py according to your needs (see preprocessing data above)

4. Run scripts.py 
   


## Output Description
The script provided performs the following tasks:



1. Loads the training data from a CSV file (train.csv).
Calls the preprocess_data() function to preprocess the loaded data.
Splits the preprocessed data into training and testing sets using a 33% test size and a random state of 42.
Defining Neural Network Architecture:

2. Defines the sizes of the neural network layers based on the number of input features.
Instantiates neural network objects and activation functions.
Optimizing Weights and Biases:

3. Constructs a CustomTask object and defines the search space for Bayesian optimization.
Builds an optimizer using Bayesian Optimization (BO) techniques.
Optimizes weights and biases using BO with the specified number of iterations.
Evaluating on Test Data:

4. Evaluates the optimized model on the test data.
Computes the accuracy of the model on the test set.
Output Results:

5. Prints the best weights and biases obtained from optimization.
   Prints the training set accuracy obtained during optimization.
   Prints the test set accuracy obtained after evaluating the mode
