# Classification predictions with Bayesian Optimization for Multi-layer Perceptrons 

## Description
The goal of this project is to apply Bayesian Optimization for Multi-layer Perceptrons architecture to a Feed Forward Neural Network as an alternative to traditional back-propagation.

## Project Structure
- `BO_RandomDecomposition_NeuralNetworks/HEBO/scripts`: Directory containing Bo_usage.py taking the Titanic dataset as input.
  - `config_NN_methods.py`
  - `BO_methods.py`
  - `BO_pipeline.py`

### Preprocessing Data
This project includes a preprocessing function designed specifically for the Titanic dataset from Kaggle. The function handles missing data and encodes categorical variables, such as the 'Sex' column.

**Note:** If you are using the Titanic dataset or a similar dataset with the same structure, you can directly utilize this function for preprocessing. If the features of your dataset vary, you may need to customize or develop additional preprocessing methods to suit the characteristics and requirements of your data. You can refer to the preprocessing function as a reference or starting point for building your own preprocessing pipeline.

## Usage

### Setting up the Environment
1. Install required packages, create and activate the conda environment called mcbo_env by running the following commands:
   ```bash
   conda env create -f environment.yml
   conda activate mcbo_env
If your conda environment has been previously created and/or you need to include additional packages in environment.yml, run instead:
   ```bash
   conda env update --file environment.yml --name mcbo_env 
```
2. Assure the mcbo_env interpreter is activated.


## Getting Started
- Clone the repository:
   ```bash
   git clone https://github.com/y-prog/BO_RandomDecomposition_NeuralNetworks.git
   ```
  
- Open Bo_usage.py in the directory BO_RandomDecomposition_NeuralNetworks/HEBO/scripts.

- Configure Bo_usage.py according to your needs (see preprocessing data above).

- Run Bo_usage.py.

## Output Description
The script provided performs the following tasks:

- Loads the training data from a CSV file (`train.csv`).
- Calls the `preprocess_data()` function to preprocess the loaded data.
- Splits the preprocessed data into training and testing sets using a 33% test size and a random state of 42.
- Defines the sizes of the neural network layers based on the number of input features.
- Instantiates neural network objects and activation functions.
- Constructs a CustomTask object and defines the search space for Bayesian optimization.
- Builds an optimizer using Bayesian Optimization (BO) techniques.
- Optimizes weights and biases using BO with the specified number of iterations.
- Evaluates the optimized model on the test data.
- Computes the accuracy of the model on the test set.
- Prints the best weights and biases obtained from optimization.
- Prints the training set accuracy obtained during optimization.
- Prints the test set accuracy obtained after evaluating the model.


