# Classification predictions with Bayesian Optimization for Multi-layer Perceptrons 

## Description
The goal of this project is to apply Bayesian Optimization for Multi-layer Perceptrons architecture to a Feed Forward Neural Network as an alternative to traditional back-propagation.

## Project Structure
- `BO_RandomDecomposition_NeuralNetworks/HEBO/scripts`: Directory containing Bo_usage.py taking our datasets preprocessing method as input together with the number of iterations.
  - `config_NN_methods.py`
  - `BO_methods.py`
  - `BO_pipeline.py`
  - `proj_methods.py`
  - `preprocessors.py`

### Preprocessing Data
The file called `preprocessors.py` includes preprocessing functions specifically designed for the four investigated datasets. Depending upon which dataset, its preprocessing function might include handling of the missing data, normalization, feature engineering and encoding; such methods are stored and imported from `proj_methods.py`.

**Note:** If the user would like to test other dataset, the preprocessing method would most likely have to be rewritten. If the features of your dataset vary, you may need to customize or develop additional preprocessing methods to suit the characteristics and requirements of your data. You can refer to the preprocessing function as a reference or starting point for building your own preprocessing pipeline.

**Note:** Due to the size of `creditcard.csv`, its url has been included in the datasets folder together with the rest of the csv files. In order for this file to be tested, the user will have to manually download it from the provided url and drag it into the dataset folder 



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

- Bo_usage.py imports different preprocessing methods discussed in section 1 above.

- Run Bo_usage.py.

## Output Description
The script provided performs the following tasks:

- Loads the training data from a CSV file (`train.csv`).
- Calls the `preprocess_data()` function to preprocess the loaded data.
- Splits the preprocessed data into training and testing sets using a 25% test size and a random state of 42.
- Defines the sizes of the neural network layers based on the number of input features.
- Instantiates neural network objects and activation functions.
- Constructs a CustomTask object and defines the search space for Bayesian optimization.
- Builds an optimizer using Bayesian Optimization (BO) techniques.
- Optimizes weights and biases using BO with the specified number of iterations.
- Computes the accuracy of the model on the train test set.
- Prints both Bayesian Optimization and Back Propagation train and test results 

