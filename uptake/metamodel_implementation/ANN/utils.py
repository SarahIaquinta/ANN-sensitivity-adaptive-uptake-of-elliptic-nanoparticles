import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import openturns as ot

def export_predictions(y_test, y_pred, Q2, complete_filename):
    """
    Exports the objects to the metamodel .pkl

    Parameters:
        ----------
        y_test: array
            test data used to validate the ANN
        y_pred: array
            predictions of the ANN
        Q2: float
            predictivity factor obtained with y_test and y_pred, computed with sklearn

    Returns:
        -------
        None
    """
    with open(complete_filename, "wb") as f:
        pickle.dump(
            [y_test, y_pred, Q2],
            f,
        )

def export_gridsearch(scaler, grid, complete_filename):
    """
    Exports the objects to the metamodel .pkl

    Parameters:
        ----------
        y_test: array
            test data used to validate the ANN
        y_pred: array
            predictions of the ANN
        Q2: float
            predictivity factor obtained with y_test and y_pred, computed with sklearn

    Returns:
        -------
        None
    """
    with open(complete_filename, "wb") as f:
        pickle.dump(
            [scaler, grid],
            f,
        )

def extract_gridsearch(complete_filename):
    with open(complete_filename, "rb") as f:
        [scaler, grid] = pickle.load(f)
    return scaler, grid

def compute_output_ANN(grid, input):
    sc_X = StandardScaler()
    X_testscaled=sc_X.fit_transform(input)
    output_ANN_before_reshape = grid.predict(X_testscaled)
    (size_output_ANN_before_reshape,) = np.shape(output_ANN_before_reshape)
    output_ANN = ot.Sample(size_output_ANN_before_reshape, 1)
    input_ANN = ot.Sample(size_output_ANN_before_reshape, 6)
    for i in range(size_output_ANN_before_reshape):
        output_ANN[i, 0] = output_ANN_before_reshape[i]
        for j in range(6):
            input_ANN[i, j] = X_testscaled[i, j]
    return input_ANN, output_ANN

def compute_output_ANN(x):
    complete_filename_grid = "grid_search.pkl"
    sc_X, grid = extract_gridsearch(complete_filename_grid)
    # sc_X = StandardScaler()
    input = ot.Sample(1, 6)
    output_reshaped = ot.Sample(1, 1)
    for i in range(6):
        input[0, i] = x[i]
    # input[0, 0] = x
    X_testscaled=sc_X.transform(input)
    output = grid.predict(X_testscaled)
    output_reshaped[0, 0] = output[0]
    # print(output_reshaped)
    y = [output[0]]
    # print('hello')
    return y

def compute_output_ANN_new_settings(x):
    complete_filename_grid = "grid_search_new_settings.pkl"
    sc_X, grid = extract_gridsearch(complete_filename_grid)
    # sc_X = StandardScaler()
    input = ot.Sample(1, 6)
    output_reshaped = ot.Sample(1, 1)
    for i in range(6):
        input[0, i] = x[i]
    # input[0, 0] = x
    X_testscaled=sc_X.transform(input)
    output = grid.predict(X_testscaled)
    output_reshaped[0, 0] = output[0]
    # print(output_reshaped)
    y = [output[0]]
    # print('hello')
    return y


def extract_predictions(complete_filename):
    with open(complete_filename, "rb") as f:
        [y_test, y_pred, Q2] = pickle.load(f)
    return y_test, y_pred, Q2

def export_settings_for_metamodels(datapresetting, training_amount, shuffled_sample, X_train, X_test, y_train, y_test):
    size, _ = np.shape(shuffled_sample)
    rough_size = int(size // 1000)
    complete_filename = "settings_for_comparison_surrogates_size" + str(rough_size) + ".pkl"
    with open(complete_filename, "wb") as f:
        pickle.dump([datapresetting, training_amount, shuffled_sample, X_train, X_test, y_train, y_test],
            f,
        )

def export_settings_for_metamodels_new_settings(datapresetting, training_amount, shuffled_sample, X_train, X_test, y_train, y_test):
    size, _ = np.shape(shuffled_sample)
    rough_size = int(size // 1000)
    complete_filename = "settings_for_comparison_surrogates_new_settings_size" + str(rough_size) + ".pkl"
    with open(complete_filename, "wb") as f:
        pickle.dump([datapresetting, training_amount, shuffled_sample, X_train, X_test, y_train, y_test],
            f,
        )


def extract_settings_for_comparison_surrogates(size):
    complete_filename = "settings_for_comparison_surrogates_size" + str(size) + ".pkl"
    with open(complete_filename, "rb") as f:
        [datapresetting, training_amount, shuffled_sample, X_train, X_test, y_train, y_test] = pickle.load(f)
    return datapresetting, training_amount, shuffled_sample, X_train, X_test, y_train, y_test

def extract_settings_for_comparison_surrogates_new_settings(size):
    complete_filename = "settings_for_comparison_surrogates_new_settings_size" + str(size) + ".pkl"
    with open(complete_filename, "rb") as f:
        [datapresetting, training_amount, shuffled_sample, X_train, X_test, y_train, y_test] = pickle.load(f)
    return datapresetting, training_amount, shuffled_sample, X_train, X_test, y_train, y_test