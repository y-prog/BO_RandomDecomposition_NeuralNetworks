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

To run the main script:

1. Clone the repository:
   
   git clone https://github.com/y-prog/BO_RandomDecomposition_NeuralNetworks.git

2. open scripts.py in the following directory BO_RandomDecomposition_NeuralNetworks
/HEBO/scripts

3. configure scripts.py according to your needs (see preprocessing data above)

4. Run scripts.py 
   
