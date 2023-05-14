import pickle
from pathlib import Path

import openturns as ot
import numpy as np

def create_pkl_name(type_of_metamodel, training_amount, folder=""):
    """
    Creates the name of the .pkl file in which the metamodel will be stored

    Parameters:
        ----------
        type_of_metamodel: string
            name of the metamodel that has been computed.
            Possible values :
                "Kriging"
        training_amount: float (between 0 and 1)
            amount of the data that is used to train the metamodel
        folder: string
            the name of the folder in which the .pkl will be created and saved
            default value: ""

    Returns:
        -------
        complete_filename: str
            name of the .pkl file

    """
    path = Path.cwd() / folder
    pkl_name = "metamodel_" + type_of_metamodel + "_trainingamount_" + str(training_amount) + ".pkl"
    complete_filename = path / pkl_name
    return complete_filename


def create_pkl_name_sensitivityalgo(
    type_of_metamodel,
    training_amount,
    experiment_size,
    sobol_implementation,
    folder="sensitivity_analysis",
):
    """
    Creates the name of the .pkl file in which the sensitivity
        algorithm will be stored

    Parameters:
        ----------
        type_of_metamodel: string
            name of the metamodel that has been computed.
            Possible values :
                "Kriging"
        training_amount: float (between 0 and 1)
            amount of the data that is used to train the metamodel
        experiment_size: float
            number of simulations used to compute the sensitivity algorithm
        sobol_implementation: string
            name of the Sobol algorithm implemented
            Possible values :
                "Jansen", "Martinez", "MauntzKucherenko", "Saltelli"
        folder: string
            the name of the folder in which the .pkl will be created and saved
            default value: "sensitivity_analysis"

    Returns:
        -------
        complete_filename: str
            name of the .pkl file

    """
    path = Path.cwd() / folder
    pkl_name = (
        "sensitivityalgo="
        + sobol_implementation
        + "_size="
        + str(experiment_size)
        + "_metamodel="
        + type_of_metamodel
        + "_trainingamount="
        + str(training_amount)
        + ".pkl"
    )
    complete_filename = path / pkl_name
    return complete_filename


def extract_metamodel_and_data_from_pkl(complete_filename):
    """
    Extracts the objects that have been stored with the metamodel .pkl

    Parameters:
        ----------
        complete_filename: str
            name of the .pkl file

    Returns:
        -------
        sample: ot.class
            inut dataset used to create the metamodel
        results_from_algo: ot.class
            class which possesses all the information relative to the metamodel that has been generated



    """
    with open(complete_filename, "rb") as f:
        [sample, results_from_algo] = pickle.load(f)
    return sample, results_from_algo


def export_metamodel_and_data_to_pkl(sample, results_from_algo, complete_filename):
    """
    Exports the objetcs to the metamodel .pkl

    Parameters:
        ----------
        sample: ot.class
            inut dataset used to create the metamodel
        results_from_algo: ot.class
            class which possesses all the information relative to the metamodel that has been generated
        complete_filename: str
            name of the .pkl file

    Returns:
        -------
        None
    """
    with open(complete_filename, "wb") as f:
        pickle.dump(
            [sample, results_from_algo],
            f,
        )


def export_sensitivity_algo_to_pkl(sensitivity_algo, complete_filename):
    """
    Exports the objects to the sensitivity .pkl

    Parameters:
        ----------
        sensitivity_algo: ot.class
            sensitivity algorithm
        complete_filename: str
            name of the .pkl file

    Returns:
        -------
        None
    """

    with open(complete_filename, "wb") as f:
        pickle.dump(
            [sensitivity_algo],
            f,
        )

def export_utils_sensitivity_convergence(complete_filename, Saltelli_first_order_indices_vs_experiment_size, Saltelli_total_order_indices_vs_experiment_size, Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_total_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_computation_time_vs_experiment_size, Jansen_first_order_indices_vs_experiment_size, Jansen_total_order_indices_vs_experiment_size, Jansen_first_order_indices_confidence_errorbars_vs_experiment_size, Jansen_total_order_indices_confidence_errorbars_vs_experiment_size, Jansen_computation_time_vs_experiment_size, MauntzKucherenko_first_order_indices_vs_experiment_size, MauntzKucherenko_total_order_indices_vs_experiment_size, MauntzKucherenko_first_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_total_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_computation_time_vs_experiment_size, Martinez_first_order_indices_vs_experiment_size, Martinez_total_order_indices_vs_experiment_size, Martinez_first_order_indices_confidence_errorbars_vs_experiment_size, Martinez_total_order_indices_confidence_errorbars_vs_experiment_size, Martinez_computation_time_vs_experiment_size):
    with open(complete_filename, "wb") as f:
        pickle.dump(
            [Saltelli_first_order_indices_vs_experiment_size, Saltelli_total_order_indices_vs_experiment_size, Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_total_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_computation_time_vs_experiment_size, Jansen_first_order_indices_vs_experiment_size, Jansen_total_order_indices_vs_experiment_size, Jansen_first_order_indices_confidence_errorbars_vs_experiment_size, Jansen_total_order_indices_confidence_errorbars_vs_experiment_size, Jansen_computation_time_vs_experiment_size, MauntzKucherenko_first_order_indices_vs_experiment_size, MauntzKucherenko_total_order_indices_vs_experiment_size, MauntzKucherenko_first_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_total_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_computation_time_vs_experiment_size, Martinez_first_order_indices_vs_experiment_size, Martinez_total_order_indices_vs_experiment_size, Martinez_first_order_indices_confidence_errorbars_vs_experiment_size, Martinez_total_order_indices_confidence_errorbars_vs_experiment_size, Martinez_computation_time_vs_experiment_size],
            f,
        )    

def export_utils_sensitivity_convergence_confinterval(complete_filename, Saltelli_first_order_indices_vs_experiment_size_confinterval,Saltelli_total_order_indices_vs_experiment_size_confinterval,Jansen_first_order_indices_vs_experiment_size_confinterval,Jansen_total_order_indices_vs_experiment_size_confinterval,MauntzKucherenko_first_order_indices_vs_experiment_size_confinterval,MauntzKucherenko_total_order_indices_vs_experiment_size_confinterval,Martinez_first_order_indices_vs_experiment_size_confinterval,Martinez_total_order_indices_vs_experiment_size_confinterval):
    with open(complete_filename, "wb") as f:
        pickle.dump(
            [Saltelli_first_order_indices_vs_experiment_size_confinterval,Saltelli_total_order_indices_vs_experiment_size_confinterval,Jansen_first_order_indices_vs_experiment_size_confinterval,Jansen_total_order_indices_vs_experiment_size_confinterval,MauntzKucherenko_first_order_indices_vs_experiment_size_confinterval,MauntzKucherenko_total_order_indices_vs_experiment_size_confinterval,Martinez_first_order_indices_vs_experiment_size_confinterval,Martinez_total_order_indices_vs_experiment_size_confinterval],
            f,
        )    

def export_utils_sensitivity_convergence_gradient(complete_filename, Saltelli_first_order_indices_vs_experiment_size_gradient,Saltelli_total_order_indices_vs_experiment_size_gradient,Jansen_first_order_indices_vs_experiment_size_gradient,Jansen_total_order_indices_vs_experiment_size_gradient,MauntzKucherenko_first_order_indices_vs_experiment_size_gradient,MauntzKucherenko_total_order_indices_vs_experiment_size_gradient,Martinez_first_order_indices_vs_experiment_size_gradient,Martinez_total_order_indices_vs_experiment_size_gradient):
    with open(complete_filename, "wb") as f:
        pickle.dump(
            [Saltelli_first_order_indices_vs_experiment_size_gradient,Saltelli_total_order_indices_vs_experiment_size_gradient,Jansen_first_order_indices_vs_experiment_size_gradient,Jansen_total_order_indices_vs_experiment_size_gradient,MauntzKucherenko_first_order_indices_vs_experiment_size_gradient,MauntzKucherenko_total_order_indices_vs_experiment_size_gradient,Martinez_first_order_indices_vs_experiment_size_gradient,Martinez_total_order_indices_vs_experiment_size_gradient],
            f,
        )    


    

def extract_sensitivity_convergence(complete_filename):
    with open(complete_filename, "rb") as f:
        [Saltelli_first_order_indices_vs_experiment_size, Saltelli_total_order_indices_vs_experiment_size, Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_total_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_computation_time_vs_experiment_size, Jansen_first_order_indices_vs_experiment_size, Jansen_total_order_indices_vs_experiment_size, Jansen_first_order_indices_confidence_errorbars_vs_experiment_size, Jansen_total_order_indices_confidence_errorbars_vs_experiment_size, Jansen_computation_time_vs_experiment_size, MauntzKucherenko_first_order_indices_vs_experiment_size, MauntzKucherenko_total_order_indices_vs_experiment_size, MauntzKucherenko_first_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_total_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_computation_time_vs_experiment_size, Martinez_first_order_indices_vs_experiment_size, Martinez_total_order_indices_vs_experiment_size, Martinez_first_order_indices_confidence_errorbars_vs_experiment_size, Martinez_total_order_indices_confidence_errorbars_vs_experiment_size, Martinez_computation_time_vs_experiment_size] = pickle.load(f)
    return Saltelli_first_order_indices_vs_experiment_size, Saltelli_total_order_indices_vs_experiment_size, Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_total_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_computation_time_vs_experiment_size, Jansen_first_order_indices_vs_experiment_size, Jansen_total_order_indices_vs_experiment_size, Jansen_first_order_indices_confidence_errorbars_vs_experiment_size, Jansen_total_order_indices_confidence_errorbars_vs_experiment_size, Jansen_computation_time_vs_experiment_size, MauntzKucherenko_first_order_indices_vs_experiment_size, MauntzKucherenko_total_order_indices_vs_experiment_size, MauntzKucherenko_first_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_total_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_computation_time_vs_experiment_size, Martinez_first_order_indices_vs_experiment_size, Martinez_total_order_indices_vs_experiment_size, Martinez_first_order_indices_confidence_errorbars_vs_experiment_size, Martinez_total_order_indices_confidence_errorbars_vs_experiment_size, Martinez_computation_time_vs_experiment_size



def extract_sensitivity_convergence_confinterval(complete_filename):
    with open(complete_filename, "rb") as f:
        [Saltelli_first_order_indices_vs_experiment_size_confinterval,Saltelli_total_order_indices_vs_experiment_size_confinterval,Jansen_first_order_indices_vs_experiment_size_confinterval,Jansen_total_order_indices_vs_experiment_size_confinterval,MauntzKucherenko_first_order_indices_vs_experiment_size_confinterval,MauntzKucherenko_total_order_indices_vs_experiment_size_confinterval,Martinez_first_order_indices_vs_experiment_size_confinterval,Martinez_total_order_indices_vs_experiment_size_confinterval] = pickle.load(f)
    return Saltelli_first_order_indices_vs_experiment_size_confinterval,Saltelli_total_order_indices_vs_experiment_size_confinterval,Jansen_first_order_indices_vs_experiment_size_confinterval,Jansen_total_order_indices_vs_experiment_size_confinterval,MauntzKucherenko_first_order_indices_vs_experiment_size_confinterval,MauntzKucherenko_total_order_indices_vs_experiment_size_confinterval,Martinez_first_order_indices_vs_experiment_size_confinterval,Martinez_total_order_indices_vs_experiment_size_confinterval

def extract_sensitivity_convergence_gradient(complete_filename):
    with open(complete_filename, "rb") as f:
        [Saltelli_first_order_indices_vs_experiment_size_gradient,Saltelli_total_order_indices_vs_experiment_size_gradient,Jansen_first_order_indices_vs_experiment_size_gradient,Jansen_total_order_indices_vs_experiment_size_gradient,MauntzKucherenko_first_order_indices_vs_experiment_size_gradient,MauntzKucherenko_total_order_indices_vs_experiment_size_gradient,Martinez_first_order_indices_vs_experiment_size_gradient,Martinez_total_order_indices_vs_experiment_size_gradient] = pickle.load(f)
    return Saltelli_first_order_indices_vs_experiment_size_gradient,Saltelli_total_order_indices_vs_experiment_size_gradient,Jansen_first_order_indices_vs_experiment_size_gradient,Jansen_total_order_indices_vs_experiment_size_gradient,MauntzKucherenko_first_order_indices_vs_experiment_size_gradient,MauntzKucherenko_total_order_indices_vs_experiment_size_gradient,Martinez_first_order_indices_vs_experiment_size_gradient,Martinez_total_order_indices_vs_experiment_size_gradient



def extract_sensitivity_algo_from_pkl(complete_filename):
    """
    Extracts the objects from the sensitivity .pkl

    Parameters:
        ----------
        complete_filename: str
            name of the .pkl file

    Returns:
        -------
        sensitivity_algo: ot.class
            sensitivity algorithm
    """
    with open(complete_filename, "rb") as f:
        [sensitivity_algo] = pickle.load(f)
    return sensitivity_algo


def rescale_sample(vector):
    vector_start0 = [k[0] - vector.getMin()[0] for k in vector]
    vector_end2 = [k * 2 / max(vector_start0) for k in vector_start0]
    vector_normalized = [k - 1 for k in vector_end2]
    return vector_normalized


def transform_vector_to_Sample(vector):
    sample = ot.Sample(len(vector), 1)
    for k in range(len(vector)):
        sample[k, 0] = vector[k]
    return sample

def compute_distance(sample1, sample2):
    dimension = np.shape(sample1)[0]
    distance = np.sqrt(np.sum([(sample1[i] - sample2[i])**2 for i in range(dimension)]))
    return distance

def export_variogram(filename, distances, y_distances):
    with open(filename, "wb") as f:
            pickle.dump(
                [distances, y_distances],
                f,
            )

def extract_variogram(filename):
    with open(filename, "rb") as f:
        [distances, y_distances] = pickle.load(f)
    return distances, y_distances

def squared_exponential(distance, theta, sigma):
    h = distance / theta
    C = (sigma**2) * np.exp(-0.5 * h**2)
    return C

def fit_variogram_squared_exponential(distances, theta, sigma):
    fit = np.zeros_like(distances)
    for i in range(len(distances)):
        C = squared_exponential(distances[i], theta, sigma)
        fit[i] = C
    return fit

def spherical(distance, theta, sigma, a):
    h = np.abs(distance / theta)
    C = (sigma**2) * (1 - 0.5 * h / a * (3 - (h/a)**2))
    return C

def fit_variogram_spherical(distances, theta, sigma, a):
    fit = np.zeros_like(distances)
    for i in range(len(distances)):
        C = spherical(distances[i], theta, sigma, a)
        fit[i] = C
    return fit
