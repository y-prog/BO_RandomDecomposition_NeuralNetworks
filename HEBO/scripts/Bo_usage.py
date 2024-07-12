from data_preprocessors import data_preprocessor
from proj_methods import NN_optimizer

titanic_preprocessor = data_preprocessor().preprocess_titanic_data
creditcards_preprocessor = data_preprocessor().preprocess_creditcards_data
realestate_preprocessor = data_preprocessor().preprocess_realestate_data
census_preprocessor = data_preprocessor().preprocess_census_data

def main(nr_iter, preprocessing_fn):
    for i in range(nr_iter):
        print(f'iter nr {i + 1}')
        X_train, X_test, y_train, y_test = preprocessing_fn()  # Call the function
        nn_opt = NN_optimizer(X_train, X_test, y_train, y_test)
        layer_sizes_bo = [X_train.shape[1], 8, 1]  # nn layers for bayesian opt
        bo_acc = nn_opt.optimize_bo(layer_sizes_bo, 300, 50)
        bp_acc = nn_opt.build_and_train_nnbp(50, 400)
        print(f'Train accuracy BO: {bo_acc[0]:.4f}, Test accuracy BO: {bo_acc[1]:.4f}')
        print(f'Train accuracy BP: {bp_acc[0]:.4f}, Test accuracy BP: {bp_acc[1]:.4f}')


if __name__ == '__main__':
    main(5, titanic_preprocessor)

