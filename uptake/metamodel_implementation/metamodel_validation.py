import pickle
from tkinter import S

import numpy as np
import openturns as ot
import openturns.viewer as viewer
import scipy
from scipy.interpolate import UnivariateSpline
from sklearn.neighbors import KernelDensity

ot.Log.Show(ot.Log.NONE)

import seaborn as sns
from matplotlib import pylab as plt

import uptake.metamodel_implementation.utils as miu
from uptake.figures.utils import CreateFigure, Fonts, SaveFigure, XTickLabels, XTicks
from uptake.metamodel_implementation.metamodel_creation import DataPreSetting, MetamodelPostTreatment


class MetamodelValidation:
    """A class that contains the methods to validate a metamodel

    Attributes:
    ----------
    None

    Methods:
    -------
    validate_metamodel_with_test(self, inputTest, outputTest, metamodel):
        Constructs a metamodel validator class (from the Openturns library)
    compute_Q2(self, metamodel_validator):
        Computes the predictivity factor of the metamodel.
    """

    def __init__(self):
        """Constructs all the necessary attributes for the MetamodelValidation object.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

    def validate_metamodel_with_test(self, inputTest, outputTest, metamodel):
        """Constructs a metamodel validator class (from the Openturns library).

        Parameters:
        ----------
        inputTest: class (array)
            Part of the dataset (input variables of the model) that will be used to validate
            the metamodel
        outputTest: class (array)
            Part of the dataset (output variables of the model) that will be used to validate
            the metamodel
        metamodel: class
            metamodel object (from the OpenTurns library)

        Returns:
        -------
        metamodel_validator: OT class
            Tool from the Openturns library used to validate a metamodel
        """

        metamodel_validator = ot.MetaModelValidation(inputTest, outputTest, metamodel)
        return metamodel_validator

    def compute_Q2(self, metamodel_validator):
        """Computes the predictivity factor of the metamodel.

        Parameters:
        ----------
        metamodel_validator: class
            Tool from the Openturns library used to validate a metamodel

        Returns:
        -------
        Q2: class (array)
            Predictivity factor
        """

        Q2 = metamodel_validator.computePredictivityFactor()
        return Q2

    def plot_prediction_vs_true_value_manual(
        self, type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    ):
        predicted_output = metamodel(inputTest)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        palette = sns.color_palette("Paired")
        orange = palette[-5]
        purple = palette[-3]
        color_plot = orange
        if type_of_metamodel == "Kriging":
            color_plot = purple
        ax.plot([0.05, 0.8], [0.05, 0.8], "-k", linewidth=2)
        ax.plot(outputTest, predicted_output, "o", color=color_plot)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ax.set_xticklabels(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ax.set_yticklabels(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlim((0, 0.83))
        ax.set_ylim((0, 0.83))
        ax.set_xlabel(r"true values of $\Psi_3$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"predicted values of $\Psi_3$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        savefigure.save_as_png(fig, type_of_metamodel + "_circular_" + str(pixels))

    def plot_prediction_vs_true_value_manual_mechanoadaptation_vs_passive_circular(
        self, type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    ):
        predicted_output = metamodel(inputTest)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        palette = sns.color_palette("Paired")
        orange = palette[-5]
        purple = palette[-3]
        color_plot = orange
        if type_of_metamodel == "Kriging":
            color_plot = purple
        ax.plot(outputTest, predicted_output, "o", color=color_plot)
        ax.plot([0, 1], [0, 1], "-k", linewidth=2)
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
        ax.set_xticklabels(
            [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
        ax.set_yticklabels(
            [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlim((-0.22, 1.22))
        ax.set_ylim((-0.22, 1.22))
        ax.grid(linestyle="--")
        ax.set_xlabel(r"true values of $\tilde{F}$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"predicted values of $\tilde{F}$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        savefigure.save_as_png(fig, type_of_metamodel + "_mechanoadaptation_vs_passive_circular_" + str(pixels))

    def plot_prediction_vs_true_value_manual_mechanoadaptation_vs_passive_elliptic(
        self, type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    ):
        predicted_output = metamodel(inputTest)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        palette = sns.color_palette("Paired")
        orange = palette[-5]
        purple = palette[-3]
        color_plot = orange
        if type_of_metamodel == "Kriging":
            color_plot = purple
        ax.plot(outputTest, predicted_output, "o", color=color_plot)
        ax.plot([0, 1], [0, 1], "-k", linewidth=2)
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
        ax.set_xticklabels(
            [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
        ax.set_yticklabels(
            [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlim((-0.02, 1.02))
        ax.set_ylim((-0.27, 1.02))
        ax.grid(linestyle="--")
        ax.set_xlabel(r"true values of $\tilde{F}$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"predicted values of $\tilde{F}$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        savefigure.save_as_png(fig, type_of_metamodel + "_mechanoadaptation_vs_passive_elliptic_" + str(pixels))

    def plot_prediction_vs_true_value_manual_mechanoadaptation_circular(
        self, type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    ):
        predicted_output = metamodel(inputTest)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        palette = sns.color_palette("Paired")
        orange = palette[-5]
        purple = palette[-3]
        color_plot = orange
        if type_of_metamodel == "Kriging":
            color_plot = purple
        ax.plot(outputTest, predicted_output, "o", color=color_plot)
        ax.plot([0, 1], [0, 1], "-k", linewidth=2)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ax.set_xticklabels(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ax.set_yticklabels(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlim((0, 0.83))
        ax.set_ylim((0, 0.83))
        ax.grid(linestyle="--")
        ax.set_xlabel(r"true values of $\Psi_3$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"predicted values of $\Psi_3$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        savefigure.save_as_png(fig, type_of_metamodel + "_mechanoadaptation_circular_" + str(pixels))

    def plot_prediction_vs_true_value_manual_mechanoadaptation_elliptic(
        self, type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    ):
        predicted_output = metamodel(inputTest)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        palette = sns.color_palette("Paired")
        orange = palette[-5]
        purple = palette[-3]
        color_plot = orange
        if type_of_metamodel == "Kriging":
            color_plot = purple
        ax.plot(outputTest, predicted_output, "o", color=color_plot)
        ax.plot([0, 1], [0, 1], "-k", linewidth=2)
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
        ax.set_xticklabels(
            [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
        ax.set_yticklabels(
            [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlim((-0.02, 1.02))
        ax.set_ylim((-0.22, 1.02))
        ax.set_xlabel(r"true values of $\Psi_3$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"predicted values of $\Psi_3$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle="--")
        savefigure.save_as_png(fig, type_of_metamodel + "_mechanoadaptation_elliptic_" + str(pixels))

    def plot_prediction_vs_true_value_manual_constant_elliptic(
        self, type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    ):
        predicted_output = metamodel(inputTest)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        palette = sns.color_palette("Paired")
        orange = palette[-5]
        purple = palette[-3]
        color_plot = orange
        if type_of_metamodel == "Kriging":
            color_plot = purple
        ax.plot(outputTest, predicted_output, "o", color=color_plot)
        ax.plot([0, 1], [0, 1], "-k", linewidth=2)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xticklabels(
            [0, 0.2, 0.4, 0.6, 0.8, 1],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels(
            [0, 0.2, 0.4, 0.6, 0.8, 1],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlim((-0.02, 1.02))
        ax.set_ylim((-0.08, 1.02))
        ax.grid(linestyle="--")
        ax.set_xlabel(r"true values of $\tilde{F}$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"predicted values of $\tilde{F}$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        savefigure.save_as_png(fig, type_of_metamodel + "_constant_elliptic_" + str(pixels))


def metamodel_validation_routine_pce(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    degree,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "PCE"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel
    degree: int
        Dimension of the basis of polynoms


    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(type_of_metamodel + str(degree), training_amount)
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data(shuffled_sample)

    gamma_bar_r_list_rescaled = miu.rescale_sample(inputTest[:, 0])
    gamma_bar_fs_bar_list_rescaled = miu.rescale_sample(inputTest[:, 1])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(inputTest[:, 2])
    input_sample_Test_rescaled = ot.Sample(len(gamma_bar_r_list_rescaled), 3)
    for k in range(len(gamma_bar_r_list_rescaled)):
        input_sample_Test_rescaled[k, 0] = gamma_bar_r_list_rescaled[k]
        input_sample_Test_rescaled[k, 1] = gamma_bar_fs_bar_list_rescaled[k]
        input_sample_Test_rescaled[k, 2] = gamma_bar_lambda_list_rescaled[k]

    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(
        input_sample_Test_rescaled, outputTest, metamodel
    )
    metamodelvalidation.plot_prediction_vs_true_value_manual(
        type_of_metamodel, input_sample_Test_rescaled, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )
    Q2 = metamodel_validator.computePredictivityFactor()
    residual, relative_error = metamodelposttreatment.get_errors_from_metamodel(results_from_algo)
    return Q2, residual, relative_error


def metamodel_validation_routine_pce_constant_elliptic(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    degree,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "PCE"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel
    degree: int
        Dimension of the basis of polynoms


    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(type_of_metamodel + "_constant_elliptic" + str(degree), training_amount)
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data_constant_elliptic(shuffled_sample)

    gamma_bar_0_list_rescaled = miu.rescale_sample(inputTest[:, 0])
    sigma_bar_list_rescaled = miu.rescale_sample(inputTest[:, 1])
    r_bar_list_rescaled = miu.rescale_sample(inputTest[:, 2])
    input_sample_Test_rescaled = ot.Sample(len(gamma_bar_0_list_rescaled), 3)
    for k in range(len(gamma_bar_0_list_rescaled)):
        input_sample_Test_rescaled[k, 0] = gamma_bar_0_list_rescaled[k]
        input_sample_Test_rescaled[k, 1] = sigma_bar_list_rescaled[k]
        input_sample_Test_rescaled[k, 2] = r_bar_list_rescaled[k]

    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(
        input_sample_Test_rescaled, outputTest, metamodel
    )
    metamodelvalidation.plot_prediction_vs_true_value_manual_constant_elliptic(
        type_of_metamodel, input_sample_Test_rescaled, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )
    Q2 = metamodel_validator.computePredictivityFactor()
    residual, relative_error = metamodelposttreatment.get_errors_from_metamodel(results_from_algo)
    return Q2, residual, relative_error


def metamodel_validation_routine_pce_mechanoadaptation_vs_passive_elliptic(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    degree,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "PCE"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel
    degree: int
        Dimension of the basis of polynoms


    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(
        type_of_metamodel + "_mechanoadaptation_vs_passive_elliptic" + str(degree), training_amount
    )
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data_mechanoadaptation_vs_passive_elliptic(shuffled_sample)

    gamma_bar_0_list_rescaled = miu.rescale_sample(inputTest[:, 0])
    sigma_bar_list_rescaled = miu.rescale_sample(inputTest[:, 1])
    gamma_bar_r_list_rescaled = miu.rescale_sample(inputTest[:, 2])
    gamma_bar_fs_list_rescaled = miu.rescale_sample(inputTest[:, 3])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(inputTest[:, 4])
    r_bar_list_rescaled = miu.rescale_sample(inputTest[:, 5])
    input_sample_Test_rescaled = ot.Sample(len(gamma_bar_0_list_rescaled), 6)
    for k in range(len(gamma_bar_0_list_rescaled)):
        input_sample_Test_rescaled[k, 0] = gamma_bar_0_list_rescaled[k]
        input_sample_Test_rescaled[k, 1] = sigma_bar_list_rescaled[k]
        input_sample_Test_rescaled[k, 2] = gamma_bar_r_list_rescaled[k]
        input_sample_Test_rescaled[k, 3] = gamma_bar_fs_list_rescaled[k]
        input_sample_Test_rescaled[k, 4] = gamma_bar_lambda_list_rescaled[k]
        input_sample_Test_rescaled[k, 5] = r_bar_list_rescaled[k]

    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(
        input_sample_Test_rescaled, outputTest, metamodel
    )
    metamodelvalidation.plot_prediction_vs_true_value_manual_mechanoadaptation_vs_passive_elliptic(
        type_of_metamodel, input_sample_Test_rescaled, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )
    Q2 = metamodel_validator.computePredictivityFactor()
    residual, relative_error = metamodelposttreatment.get_errors_from_metamodel(results_from_algo)
    return Q2, residual, relative_error


def metamodel_validation_routine_pce_mechanoadaptation_circular(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    degree,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "PCE"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel
    degree: int
        Dimension of the basis of polynoms


    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(
        type_of_metamodel + "_mechanoadaptation_circular" + str(degree), training_amount
    )
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data_mechanoadaptation_circular(shuffled_sample)

    gamma_bar_r_list_rescaled = miu.rescale_sample(inputTest[:, 0])
    gamma_bar_fs_list_rescaled = miu.rescale_sample(inputTest[:, 1])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(inputTest[:, 2])
    input_sample_Test_rescaled = ot.Sample(len(gamma_bar_r_list_rescaled), 3)
    for k in range(len(gamma_bar_r_list_rescaled)):
        input_sample_Test_rescaled[k, 0] = gamma_bar_r_list_rescaled[k]
        input_sample_Test_rescaled[k, 1] = gamma_bar_fs_list_rescaled[k]
        input_sample_Test_rescaled[k, 2] = gamma_bar_lambda_list_rescaled[k]

    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(
        input_sample_Test_rescaled, outputTest, metamodel
    )
    metamodelvalidation.plot_prediction_vs_true_value_manual_mechanoadaptation_circular(
        type_of_metamodel, input_sample_Test_rescaled, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )
    Q2 = metamodel_validator.computePredictivityFactor()
    residual, relative_error = metamodelposttreatment.get_errors_from_metamodel(results_from_algo)
    return Q2, residual, relative_error


def metamodel_validation_routine_pce_mechanoadaptation_elliptic(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    degree,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "PCE"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel
    degree: int
        Dimension of the basis of polynoms


    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(
        type_of_metamodel + "_mechanoadaptation_elliptic" + str(degree), training_amount
    )
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data_mechanoadaptation_elliptic(shuffled_sample)

    gamma_bar_r_list_rescaled = miu.rescale_sample(inputTest[:, 0])
    gamma_bar_fs_list_rescaled = miu.rescale_sample(inputTest[:, 1])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(inputTest[:, 2])
    r_bar_list_rescaled = miu.rescale_sample(inputTest[:, 3])
    input_sample_Test_rescaled = ot.Sample(len(gamma_bar_r_list_rescaled), 4)
    for k in range(len(gamma_bar_r_list_rescaled)):
        input_sample_Test_rescaled[k, 0] = gamma_bar_r_list_rescaled[k]
        input_sample_Test_rescaled[k, 1] = gamma_bar_fs_list_rescaled[k]
        input_sample_Test_rescaled[k, 2] = gamma_bar_lambda_list_rescaled[k]
        input_sample_Test_rescaled[k, 3] = r_bar_list_rescaled[k]

    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(
        input_sample_Test_rescaled, outputTest, metamodel
    )
    metamodelvalidation.plot_prediction_vs_true_value_manual_mechanoadaptation_elliptic(
        type_of_metamodel, input_sample_Test_rescaled, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )
    Q2 = metamodel_validator.computePredictivityFactor()
    residual, relative_error = metamodelposttreatment.get_errors_from_metamodel(results_from_algo)
    return Q2, residual, relative_error


def metamodel_validation_routine_pce_mechanoadaptation_vs_passive_circular(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    degree,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "PCE"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel
    degree: int
        Dimension of the basis of polynoms


    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(
        type_of_metamodel + "_mechanoadaptation_vs_passive_circular" + str(degree), training_amount
    )
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data_mechanoadaptation_vs_passive_circular(shuffled_sample)

    gamma_bar_0_list_rescaled = miu.rescale_sample(inputTest[:, 0])
    sigma_bar_list_rescaled = miu.rescale_sample(inputTest[:, 1])
    gamma_bar_r_list_rescaled = miu.rescale_sample(inputTest[:, 2])
    gamma_bar_fs_bar_list_rescaled = miu.rescale_sample(inputTest[:, 3])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(inputTest[:, 4])
    input_sample_Test_rescaled = ot.Sample(len(gamma_bar_r_list_rescaled), 5)
    for k in range(len(gamma_bar_r_list_rescaled)):
        input_sample_Test_rescaled[k, 0] = gamma_bar_0_list_rescaled[k]
        input_sample_Test_rescaled[k, 1] = sigma_bar_list_rescaled[k]
        input_sample_Test_rescaled[k, 2] = gamma_bar_r_list_rescaled[k]
        input_sample_Test_rescaled[k, 3] = gamma_bar_fs_bar_list_rescaled[k]
        input_sample_Test_rescaled[k, 4] = gamma_bar_lambda_list_rescaled[k]

    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(
        input_sample_Test_rescaled, outputTest, metamodel
    )
    metamodelvalidation.plot_prediction_vs_true_value_manual_mechanoadaptation_vs_passive_circular(
        type_of_metamodel, input_sample_Test_rescaled, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )
    Q2 = metamodel_validator.computePredictivityFactor()
    residual, relative_error = metamodelposttreatment.get_errors_from_metamodel(results_from_algo)
    return Q2, residual, relative_error


def metamodel_validation_routine_kriging(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "Kriging"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(type_of_metamodel, training_amount)
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data(shuffled_sample)

    metamodelvalidation.plot_prediction_vs_true_value_manual(
        type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )


def metamodel_validation_routine_kriging_mechanoadaptation_vs_passive_circular(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "Kriging"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(
        type_of_metamodel + "_mechanoadaptation_vs_passive_circular", training_amount
    )
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data_mechanoadaptation_vs_passive_circular(shuffled_sample)

    metamodelvalidation.plot_prediction_vs_true_value_manual_mechanoadaptation_vs_passive_circular(
        type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )
    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(inputTest, outputTest, metamodel)
    Q2 = metamodelvalidation.compute_Q2(metamodel_validator)
    print("Kriging, Q2 = ", Q2)


def metamodel_validation_routine_kriging_mechanoadaptation_circular(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "Kriging"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(type_of_metamodel + "_mechanoadaptation_circular", training_amount)
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data_mechanoadaptation_circular(shuffled_sample)

    metamodelvalidation.plot_prediction_vs_true_value_manual_mechanoadaptation_circular(
        type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )
    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(inputTest, outputTest, metamodel)
    Q2 = metamodelvalidation.compute_Q2(metamodel_validator)
    print("Kriging, Q2 = ", Q2)


def metamodel_validation_routine_kriging_mechanoadaptation_elliptic(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "Kriging"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(type_of_metamodel + "_mechanoadaptation_elliptic", training_amount)
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data_mechanoadaptation_elliptic(shuffled_sample)

    metamodelvalidation.plot_prediction_vs_true_value_manual_mechanoadaptation_elliptic(
        type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )
    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(inputTest, outputTest, metamodel)
    Q2 = metamodelvalidation.compute_Q2(metamodel_validator)
    print("Kriging, Q2 = ", Q2)


def metamodel_validation_routine_kriging_mechanoadaptation_vs_passive_elliptic(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "Kriging"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(
        type_of_metamodel + "_mechanoadaptation_vs_passive_elliptic", training_amount
    )
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data_mechanoadaptation_vs_passive_elliptic(shuffled_sample)

    metamodelvalidation.plot_prediction_vs_true_value_manual_mechanoadaptation_vs_passive_elliptic(
        type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )
    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(inputTest, outputTest, metamodel)
    Q2 = metamodelvalidation.compute_Q2(metamodel_validator)
    print("Kriging, Q2 = ", Q2)


def metamodel_validation_routine_kriging_constant_elliptic(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "Kriging"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(type_of_metamodel + "_constant_elliptic", training_amount)
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data_constant_elliptic(shuffled_sample)

    metamodelvalidation.plot_prediction_vs_true_value_manual_constant_elliptic(
        type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )
    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(inputTest, outputTest, metamodel)
    Q2 = metamodelvalidation.compute_Q2(metamodel_validator)
    print("Kriging, Q2 = ", Q2)


def plot_error_vs_degree_pce(
    degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
):
    """Plots the LOO error and the predictivity factor of the PCE with respect to its
        truncature degree

    Parameters:
    ----------
    degree_list: list
        List of the truncature degrees to be investigated
    Q2_list: list
        List of the predictivity factors obtained for each degree from degree_list
    relativeerror_list: list
        List of the LOO errors obtained for each degree from degree_list

    Returns:
    -------
    None
    """

    palette = sns.color_palette("Set2")
    vert_clair, orange = palette[0], palette[1]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    ax.plot(degree_list, Q2_list, color=vert_clair, label="$Q_2$", linewidth=3)
    ax.plot(degree_list, relativeerror_list, color=orange, label=r"$\epsilon_{LOO}$", linewidth=3)
    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    plt.plot(degree_list[max_Q2_index], max_Q2, "o", color="k", markersize=15, mfc="none")
    min_relativeerror = min(relativeerror_list)
    min_relativeerror_index = relativeerror_list.index(min_relativeerror)
    plt.plot(degree_list[min_relativeerror_index], min_relativeerror, "o", color="k", markersize=15, mfc="none")
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_xticklabels(
        [1, 10, 20, 30, 40, 50],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(
        [0, 0.2, 0.4, 0.6, 0.8, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax.set_xlim((0.5, 51))
    ax.set_ylim((-0.05, 1.05))
    ax.set_xlabel("degree $p$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$Q_2$ [ - ], $\epsilon_{LOO}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "convergence_PCE_" + str(pixels))


def plot_error_vs_degree_pce_constant_elliptic(
    degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
):
    """Plots the LOO error and the predictivity factor of the PCE with respect to its
        truncature degree

    Parameters:
    ----------
    degree_list: list
        List of the truncature degrees to be investigated
    Q2_list: list
        List of the predictivity factors obtained for each degree from degree_list
    relativeerror_list: list
        List of the LOO errors obtained for each degree from degree_list

    Returns:
    -------
    None
    """

    palette = sns.color_palette("Set2")
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    ax.plot(degree_list, Q2_list, "o", color="k", markersize=10)
    # ax.plot(degree_list, relativeerror_list, color=orange, label=r"$\epsilon_{LOO}$", linewidth=3)
    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    plt.plot(degree_list[max_Q2_index], max_Q2, "o", color="r", markersize=15, mfc="none")
    min_relativeerror = min(relativeerror_list)
    min_relativeerror_index = relativeerror_list.index(min_relativeerror)
    # plt.plot(degree_list[min_relativeerror_index], min_relativeerror, "o", color="k", markersize=15, mfc="none")
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.set_xticklabels(
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_yticklabels(
        [0.4, 0.5, 0.6, 0.7, 0.8],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    # ax.legend(prop=fonts.serif(), loc="upper left", framealpha=0.7)
    ax.set_xlim((0.5, 20.5))
    ax.set_ylim((0.4, 0.85))
    ax.grid(linestyle=":")
    ax.set_xlabel("degree $p$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$Q_2$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "convergence_PCE_constant_elliptic" + str(pixels))


def plot_error_vs_degree_pce_mechanoadaptation_vs_passive_circular(
    degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
):
    """Plots the LOO error and the predictivity factor of the PCE with respect to its
        truncature degree

    Parameters:
    ----------
    degree_list: list
        List of the truncature degrees to be investigated
    Q2_list: list
        List of the predictivity factors obtained for each degree from degree_list
    relativeerror_list: list
        List of the LOO errors obtained for each degree from degree_list

    Returns:
    -------
    None
    """

    palette = sns.color_palette("Set2")
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    plt.plot(degree_list[max_Q2_index], max_Q2, "o", color="r", markersize=15, mfc="none")
    ax.plot(degree_list, Q2_list, "o", color="k", markersize=10)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(
        [1, 2, 3, 4, 5, 6, 7],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0.75, 0.8])
    ax.set_yticklabels(
        [0.75, 0.8],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    # ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax.set_xlim((0.5, 7.5))
    ax.set_ylim((0.72, 0.805))
    ax.grid(linestyle=":")
    ax.set_xlabel("degree $p$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$Q_2$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "convergence_PCE_mechanoadaptation_vs_passive_circular" + str(pixels))


def plot_error_vs_degree_pce_mechanoadaptation_vs_passive_elliptic(
    degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
):
    """Plots the LOO error and the predictivity factor of the PCE with respect to its
        truncature degree

    Parameters:
    ----------
    degree_list: list
        List of the truncature degrees to be investigated
    Q2_list: list
        List of the predictivity factors obtained for each degree from degree_list
    relativeerror_list: list
        List of the LOO errors obtained for each degree from degree_list

    Returns:
    -------
    None
    """

    palette = sns.color_palette("Set2")
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    plt.plot(degree_list[max_Q2_index], max_Q2, "o", color="r", markersize=15, mfc="none")
    ax.plot(degree_list, Q2_list, "o", color="k", markersize=10)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(
        [1, 2, 3, 4, 5],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0.4, 0.45, 0.5, 0.55])
    ax.set_yticklabels(
        [0.4, 0.45, 0.5, 0.55],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    # ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax.set_xlim((0.5, 5.5))
    ax.set_ylim((0.38, 0.57))
    ax.grid(linestyle=":")
    ax.set_xlabel("degree $p$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$Q_2$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "convergence_PCE_mechanoadaptation_vs_passive_elliptic" + str(pixels))


def plot_error_vs_degree_pce_mechanoadaptation_circular(
    degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
):
    """Plots the LOO error and the predictivity factor of the PCE with respect to its
        truncature degree

    Parameters:
    ----------
    degree_list: list
        List of the truncature degrees to be investigated
    Q2_list: list
        List of the predictivity factors obtained for each degree from degree_list
    relativeerror_list: list
        List of the LOO errors obtained for each degree from degree_list

    Returns:
    -------
    None
    """

    palette = sns.color_palette("Set2")
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    plt.plot(degree_list[max_Q2_index], max_Q2, "o", color="r", markersize=15, mfc="none")
    ax.plot(degree_list, Q2_list, "o", color="k", markersize=10)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    ax.set_xticklabels(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0.7, 0.8, 0.9, 1])
    ax.set_yticklabels(
        [0.7, 0.8, 0.9, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    # ax.legend(prop=fonts.serif(), loc="center right", framealpha=0.7)
    ax.set_xlim((0.5, 11.5))
    ax.set_ylim((0.7, 1))
    ax.grid(linestyle=":")
    ax.set_xlabel("degree $p$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$Q_2$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "convergence_PCE_mechanoadaptation_circular_" + str(pixels))


def plot_error_vs_degree_pce_mechanoadaptation_elliptic(
    degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
):
    """Plots the LOO error and the predictivity factor of the PCE with respect to its
        truncature degree

    Parameters:
    ----------
    degree_list: list
        List of the truncature degrees to be investigated
    Q2_list: list
        List of the predictivity factors obtained for each degree from degree_list
    relativeerror_list: list
        List of the LOO errors obtained for each degree from degree_list

    Returns:
    -------
    None
    """

    palette = sns.color_palette("Set2")
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    plt.plot(degree_list[max_Q2_index], max_Q2, "o", color="r", markersize=15, mfc="none")
    ax.plot(degree_list, Q2_list, "o", color="k", markersize=10)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(
        [1, 2, 3, 4, 5, 6],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_yticklabels(
        [0.3, 0.4, 0.5, 0.6, 0.7],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((0.5, 6.5))
    ax.grid(linestyle=":")
    ax.set_xlabel("degree $p$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$Q_2$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "convergence_PCE_mechanoadaptation_elliptic_" + str(pixels))


def optimize_degree_pce(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    degree_list,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Determines the truncature degree of the PCE that maximizes the predictivity factor

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    degree_list: list
        List of the truncature degrees to be investigated
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    Q2_list = []
    residual_list = []
    relativeerror_list = []
    for degree in degree_list:
        Q2, residual, relative_error = metamodel_validation_routine_pce(
            datapresetting,
            metamodelposttreatment,
            metamodelvalidation,
            "PCE",
            training_amount,
            degree,
            createfigure,
            savefigure,
            xticks,
            pixels,
        )
        Q2_list.append(Q2[0])
        residual_list.append(residual[0])
        relativeerror_list.append(relative_error[0])

    plot_error_vs_degree_pce(
        degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
    )

    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    optimal_degree = degree_list[max_Q2_index]
    print("Optimal degree for PCE: ", int(optimal_degree))


def optimize_degree_pce_mechanoadaptation_vs_passive_elliptic(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    degree_list,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Determines the truncature degree of the PCE that maximizes the predictivity factor

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    degree_list: list
        List of the truncature degrees to be investigated
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    Q2_list = []
    residual_list = []
    relativeerror_list = []
    for degree in degree_list:
        Q2, residual, relative_error = metamodel_validation_routine_pce_mechanoadaptation_vs_passive_elliptic(
            datapresetting,
            metamodelposttreatment,
            metamodelvalidation,
            "PCE",
            training_amount,
            degree,
            createfigure,
            savefigure,
            xticks,
            pixels,
        )
        Q2_list.append(Q2[0])
        residual_list.append(residual[0])
        relativeerror_list.append(relative_error[0])

    plot_error_vs_degree_pce_mechanoadaptation_vs_passive_elliptic(
        degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
    )

    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    optimal_degree = degree_list[max_Q2_index]
    print("Optimal degree for PCE: ", int(optimal_degree), " ; Q2 = ", max_Q2)
    return int(optimal_degree)


def optimize_degree_pce_constant_elliptic(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    degree_list,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Determines the truncature degree of the PCE that maximizes the predictivity factor

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    degree_list: list
        List of the truncature degrees to be investigated
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    Q2_list = []
    residual_list = []
    relativeerror_list = []
    for degree in degree_list:
        Q2, residual, relative_error = metamodel_validation_routine_pce_constant_elliptic(
            datapresetting,
            metamodelposttreatment,
            metamodelvalidation,
            "PCE",
            training_amount,
            degree,
            createfigure,
            savefigure,
            xticks,
            pixels,
        )
        Q2_list.append(Q2[0])
        residual_list.append(residual[0])
        relativeerror_list.append(relative_error[0])

    plot_error_vs_degree_pce_constant_elliptic(
        degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
    )

    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    optimal_degree = degree_list[max_Q2_index]
    print("Optimal degree for PCE: ", int(optimal_degree), " ; Q2 = ", max_Q2)
    return int(optimal_degree)


def optimize_degree_pce_mechanoadaptation_circular(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    degree_list,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Determines the truncature degree of the PCE that maximizes the predictivity factor

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    degree_list: list
        List of the truncature degrees to be investigated
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    Q2_list = []
    residual_list = []
    relativeerror_list = []
    for degree in degree_list:
        Q2, residual, relative_error = metamodel_validation_routine_pce_mechanoadaptation_circular(
            datapresetting,
            metamodelposttreatment,
            metamodelvalidation,
            "PCE",
            training_amount,
            degree,
            createfigure,
            savefigure,
            xticks,
            pixels,
        )
        Q2_list.append(Q2[0])
        residual_list.append(residual[0])
        relativeerror_list.append(relative_error[0])

    plot_error_vs_degree_pce_mechanoadaptation_circular(
        degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
    )

    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    optimal_degree = degree_list[max_Q2_index]
    print("Optimal degree for PCE: ", int(optimal_degree), " ; Q2 = ", max_Q2)
    return int(optimal_degree)


def optimize_degree_pce_mechanoadaptation_elliptic(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    degree_list,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Determines the truncature degree of the PCE that maximizes the predictivity factor

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    degree_list: list
        List of the truncature degrees to be investigated
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    Q2_list = []
    residual_list = []
    relativeerror_list = []
    for degree in degree_list:
        Q2, residual, relative_error = metamodel_validation_routine_pce_mechanoadaptation_elliptic(
            datapresetting,
            metamodelposttreatment,
            metamodelvalidation,
            "PCE",
            training_amount,
            degree,
            createfigure,
            savefigure,
            xticks,
            pixels,
        )
        Q2_list.append(Q2[0])
        residual_list.append(residual[0])
        relativeerror_list.append(relative_error[0])

    plot_error_vs_degree_pce_mechanoadaptation_elliptic(
        degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
    )

    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    optimal_degree = degree_list[max_Q2_index]
    print("Optimal degree for PCE: ", int(optimal_degree), " ; Q2 = ", max_Q2)
    return int(optimal_degree)


def optimize_degree_pce_mechanoadaptation_vs_passive_circular(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    degree_list,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Determines the truncature degree of the PCE that maximizes the predictivity factor

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    degree_list: list
        List of the truncature degrees to be investigated
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    Q2_list = []
    residual_list = []
    relativeerror_list = []
    for degree in degree_list:
        Q2, residual, relative_error = metamodel_validation_routine_pce_mechanoadaptation_vs_passive_circular(
            datapresetting,
            metamodelposttreatment,
            metamodelvalidation,
            "PCE",
            training_amount,
            degree,
            createfigure,
            savefigure,
            xticks,
            pixels,
        )
        Q2_list.append(Q2[0])
        residual_list.append(residual[0])
        relativeerror_list.append(relative_error[0])

    plot_error_vs_degree_pce_mechanoadaptation_vs_passive_circular(
        degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
    )

    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    optimal_degree = degree_list[max_Q2_index]
    print("Optimal degree for PCE: ", int(optimal_degree), " ; Q2 = ", max_Q2)
    return int(optimal_degree)


def plot_PDF_pce_kriging_mechanoadaptation_circular(metamodelposttreatment, degree, training_amount):
    """Plots the Probability Density Functions (PDFs) of the metamodel outputs and original data

    Parameters:
    ----------
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    degree: int
        Dimension of the basis of polynoms
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    distribution_input = ot.ComposedDistribution([ot.Uniform(1, 6), ot.Uniform(-0.45, 0.45), ot.Uniform(10, 100)])
    experiment_input = ot.MonteCarloExperiment(distribution_input, int(1e5))
    sample_input_MC = experiment_input.generate()

    complete_pkl_filename_pce = miu.create_pkl_name("PCE_mechanoadaptation_circular" + str(degree), training_amount)
    shuffled_sample, results_from_algo_pce = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_pce)
    metamodel_pce = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_pce)
    complete_pkl_filename_kriging = miu.create_pkl_name("Kriging_mechanoadaptation_circular", training_amount)
    _, results_from_algo_kriging = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_kriging)
    metamodel_kriging = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_kriging)
    gamma_bar_r_list_rescaled = miu.rescale_sample(sample_input_MC[:, 0])
    gamma_bar_fs_bar_list_rescaled = miu.rescale_sample(sample_input_MC[:, 1])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(sample_input_MC[:, 2])
    input_sample_rescaled = ot.Sample(len(gamma_bar_r_list_rescaled), 3)
    for k in range(len(gamma_bar_r_list_rescaled)):
        input_sample_rescaled[k, 0] = gamma_bar_r_list_rescaled[k]
        input_sample_rescaled[k, 1] = gamma_bar_fs_bar_list_rescaled[k]
        input_sample_rescaled[k, 2] = gamma_bar_lambda_list_rescaled[k]
    input_sample = sample_input_MC  # ot.Sample(shuffled_sample[:, 0:3])
    output_model = shuffled_sample[:, -1]
    output_pce = metamodel_pce(input_sample_rescaled)
    output_kriging = metamodel_kriging(input_sample)

    # output_model_array = np.array(output_model)
    # output_pce_array = np.array(output_pce)
    # output_kriging_array = np.array(output_kriging)
    # with open('data_for_PDF_mechanoadaptation_circular.txt', 'w') as f:
    #     f.write('model \t pce \t kriging \n')
    #     for i in range(len(output_kriging_array)):
    #        f.write(str(output_model_array[i][0]) + '\t' + str(output_pce_array[i][0]) + '\t'+ str(output_kriging_array[i][0]) + '\n')
    # f.close()

    X_plot = np.linspace(0, 1, 1000)[:, None]

    kde_model = KernelDensity(kernel="gaussian", bandwidth=0.03).fit(output_model)
    log_dens_model = kde_model.score_samples(X_plot)
    kde_kriging = KernelDensity(kernel="gaussian", bandwidth=0.03).fit(output_kriging)
    log_dens_kriging = kde_kriging.score_samples(X_plot)
    kde_pce = KernelDensity(kernel="gaussian", bandwidth=0.03).fit(output_pce)
    log_dens_pce = kde_pce.score_samples(X_plot)
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_model),
        color="k",
        alpha=0.4,
        lw=4,
        linestyle="-",
        label="model",
    )
    ax.hist(shuffled_sample[:, -1], bins=20, density=True, color="lightgray", alpha=0.3, ec="black")

    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_kriging),
        color=purple,
        lw=4,
        linestyle=":",
        label="Kriging",
    )
    # ax.plot(
    #     X_plot[:, 0],
    #     np.exp(log_dens_pce),
    #     color=orange,
    #     lw=4,
    #     linestyle="--", label="PCE",
    # )
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(
        [0, 0.2, 0.4, 0.6, 0.8, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    ax.set_yticklabels(
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-0.02, 1.02))
    ax.set_xlabel(r"$\Psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$p_{\Psi_3}(\psi_3)$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax.grid(linestyle="--")
    savefigure.save_as_png(fig, "article_PDF_metamodel_mechanoadaptation_circular" + str(pixels))


def plot_PDF_pce_kriging_mechanoadaptation_elliptic(metamodelposttreatment, degree, training_amount):
    """Plots the Probability Density Functions (PDFs) of the metamodel outputs and original data

    Parameters:
    ----------
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    degree: int
        Dimension of the basis of polynoms
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    filename_qMC_mechanoadaptation_elliptic = "dataset_for_metamodel_creation_mechanoadaptation_elliptic.txt"
    datapresetting = DataPreSetting(filename_qMC_mechanoadaptation_elliptic, training_amount)
    shuffled_sample = datapresetting.shuffle_dataset_from_datafile()
    input_sample_training, _ = datapresetting.extract_training_data_from_shuffled_dataset_mechanoadaptation_elliptic(
        shuffled_sample
    )
    factory = ot.UserDefinedFactory()
    r_bar_distribution = factory.build(input_sample_training[:, 3])

    distribution_input = ot.ComposedDistribution(
        [ot.Uniform(1, 6), ot.Uniform(-0.45, 0.45), ot.Uniform(10, 100), r_bar_distribution]
    )
    experiment_input = ot.MonteCarloExperiment(distribution_input, int(1e5))
    sample_input_MC = experiment_input.generate()

    complete_pkl_filename_pce = miu.create_pkl_name("PCE_mechanoadaptation_elliptic" + str(degree), training_amount)
    shuffle_sample, results_from_algo_pce = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_pce)
    metamodel_pce = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_pce)
    complete_pkl_filename_kriging = miu.create_pkl_name("Kriging_mechanoadaptation_elliptic", training_amount)
    _, results_from_algo_kriging = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_kriging)
    metamodel_kriging = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_kriging)
    gamma_bar_r_list_rescaled = miu.rescale_sample(sample_input_MC[:1024, 0])
    gamma_bar_fs_bar_list_rescaled = miu.rescale_sample(sample_input_MC[:1024, 1])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(sample_input_MC[:1024, 2])
    r_bar_rescaled = miu.rescale_sample(sample_input_MC[:1024, 3])
    input_sample_rescaled = ot.Sample(len(gamma_bar_r_list_rescaled), 4)
    for k in range(len(gamma_bar_r_list_rescaled)):
        input_sample_rescaled[k, 0] = gamma_bar_r_list_rescaled[k]
        input_sample_rescaled[k, 1] = gamma_bar_fs_bar_list_rescaled[k]
        input_sample_rescaled[k, 2] = gamma_bar_lambda_list_rescaled[k]
        input_sample_rescaled[k, 3] = r_bar_rescaled[k]
    input_sample = sample_input_MC
    output_model = shuffled_sample[:1024, -1]
    output_pce = metamodel_pce(input_sample_rescaled)
    output_kriging = metamodel_kriging(input_sample)

    output_model_array = np.array(output_model)
    output_pce_array = np.array(output_pce)
    output_kriging_array = np.array(output_kriging)

    X_plot = np.linspace(0, 1, 1000)[:, None]

    kde_model = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_model)
    log_dens_model = kde_model.score_samples(X_plot)
    kde_kriging = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_kriging)
    log_dens_kriging = kde_kriging.score_samples(X_plot)
    kde_pce = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_pce)
    log_dens_pce = kde_pce.score_samples(X_plot)
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    ax.hist(output_model, bins=20, density=True, color="lightgray", alpha=0.3, ec="black")

    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_model),
        color="k",
        alpha=0.3,
        lw=4,
        linestyle="-",
        label="dataset",
    )
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_kriging),
        color=purple,
        lw=4,
        linestyle="--",
        label="Kriging",
    )
    # ax.plot(
    #     X_plot[:, 0],
    #     np.exp(log_dens_pce),
    #     color=orange,
    #     lw=4,
    #     linestyle=":",
    #     label="PCE",
    # )
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(
        [0, 0.2, 0.4, 0.6, 0.8, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(
        [0, 0.2, 0.4, 0.6, 0.8, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    # ax.set_ylim((0, 12.5))
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_yticklabels(
        [0, 2, 4, 6, 8, 10],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-0.02, 1.02))
    ax.set_xlabel(r"$\psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$p_{\Psi_3}(\psi_3)$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax.grid(linestyle="--")
    savefigure.save_as_png(fig, "article_PDF_metamodel_mechanoadaptation_elliptic" + str(pixels))


def plot_PDF_pce_kriging_constant_elliptic(metamodelposttreatment, degree, training_amount):
    """Plots the Probability Density Functions (PDFs) of the metamodel outputs and original data

    Parameters:
    ----------
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    degree: int
        Dimension of the basis of polynoms
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    filename_qMC_constant_elliptic = "dataset_for_metamodel_creation_feq_constant_elliptic.txt"
    datapresetting = DataPreSetting(filename_qMC_constant_elliptic, training_amount)
    shuffled_sample = datapresetting.shuffle_dataset_from_datafile()
    input_sample_training, _ = datapresetting.extract_training_data_from_shuffled_dataset_constant_elliptic(
        shuffled_sample
    )
    factory = ot.UserDefinedFactory()
    r_bar_distribution = factory.build(input_sample_training[:, 2])

    distribution_input = ot.ComposedDistribution([ot.Uniform(1, 8), ot.Uniform(0.5, 5.5), r_bar_distribution])
    experiment_input = ot.MonteCarloExperiment(distribution_input, int(1e5))
    sample_input_MC = experiment_input.generate()

    complete_pkl_filename_pce = miu.create_pkl_name("PCE_constant_elliptic" + str(degree), training_amount)
    shuffled_sample, results_from_algo_pce = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_pce)
    metamodel_pce = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_pce)
    complete_pkl_filename_kriging = miu.create_pkl_name("Kriging_constant_elliptic", training_amount)
    _, results_from_algo_kriging = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_kriging)
    metamodel_kriging = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_kriging)
    gamma_bar_0_list_rescaled = miu.rescale_sample(sample_input_MC[:, 0])
    sigma_bar_list_rescaled = miu.rescale_sample(sample_input_MC[:, 1])
    r_bar_list_rescaled = miu.rescale_sample(sample_input_MC[:, 2])
    input_sample_rescaled = ot.Sample(len(gamma_bar_0_list_rescaled), 3)
    for k in range(len(gamma_bar_0_list_rescaled)):
        input_sample_rescaled[k, 0] = gamma_bar_0_list_rescaled[k]
        input_sample_rescaled[k, 1] = sigma_bar_list_rescaled[k]
        input_sample_rescaled[k, 2] = r_bar_list_rescaled[k]
    input_sample = sample_input_MC  # ot.Sample(shuffled_sample[:, 0:3])
    output_model = shuffled_sample[:, -2]
    output_pce = metamodel_pce(input_sample_rescaled)
    output_kriging = metamodel_kriging(input_sample)

    X_plot = np.linspace(0, 1, 1000)[:, None]

    kde_model = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_model)
    log_dens_model = kde_model.score_samples(X_plot)
    kde_kriging = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_kriging)
    log_dens_kriging = kde_kriging.score_samples(X_plot)
    kde_pce = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_pce)
    log_dens_pce = kde_pce.score_samples(X_plot)
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_model),
        color="k",
        alpha=0.4,
        lw=4,
        linestyle="-",
        label="model",
    )
    ax.hist(shuffled_sample[:, -2], bins=20, density=True, color="lightgray", alpha=0.3, ec="black")

    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_kriging),
        color=purple,
        lw=4,
        linestyle=":",
        label="Kriging",
    )
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_pce),
        color=orange,
        lw=4,
        linestyle="--",
        label="PCE",
    )
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(
        [0, 0.2, 0.4, 0.6, 0.8, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 2, 4, 6])
    ax.set_yticklabels(
        [0, 2, 4, 6],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-0.02, 1.02))
    ax.set_xlabel(r"$\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$p_{\tilde{F}}(\tilde{f})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax.grid(linestyle="--")
    savefigure.save_as_png(fig, "PDF_metamodel_constant_elliptic" + str(pixels))


def plot_PDF_pce_kriging_mechanoadaptation_vs_passive_elliptic(metamodelposttreatment, degree, training_amount):
    """Plots the Probability Density Functions (PDFs) of the metamodel outputs and original data

    Parameters:
    ----------
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    degree: int
        Dimension of the basis of polynoms
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    filename_qMC_mechanoadaptation_vs_passive_elliptic = (
        "dataset_for_metamodel_creation_mechanoadaptation_vs_passive_elliptic.txt"
    )
    datapresetting = DataPreSetting(filename_qMC_mechanoadaptation_vs_passive_elliptic, training_amount)
    shuffled_sample = datapresetting.shuffle_dataset_from_datafile()
    (
        input_sample_training,
        _,
    ) = datapresetting.extract_training_data_from_shuffled_dataset_mechanoadaptation_vs_passive_elliptic(
        shuffled_sample
    )
    factory = ot.UserDefinedFactory()
    r_bar_distribution = factory.build(input_sample_training[:, 5])

    distribution_input = ot.ComposedDistribution(
        [
            ot.Uniform(1, 8),
            ot.Uniform(0.5, 5.5),
            ot.Uniform(1, 6),
            ot.Uniform(-0.45, 0.45),
            ot.Uniform(10, 100),
            r_bar_distribution,
        ]
    )
    experiment_input = ot.MonteCarloExperiment(distribution_input, int(1e5))
    sample_input_MC = experiment_input.generate()

    complete_pkl_filename_pce = miu.create_pkl_name(
        "PCE_mechanoadaptation_vs_passive_elliptic" + str(degree), training_amount
    )
    shuffled_sample, results_from_algo_pce = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_pce)
    metamodel_pce = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_pce)
    complete_pkl_filename_kriging = miu.create_pkl_name(
        "Kriging_mechanoadaptation_vs_passive_elliptic", training_amount
    )
    _, results_from_algo_kriging = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_kriging)
    metamodel_kriging = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_kriging)

    gamma_bar_0_list_rescaled = miu.rescale_sample(sample_input_MC[:, 0])
    sigma_bar_list_rescaled = miu.rescale_sample(sample_input_MC[:, 1])
    gamma_bar_r_list_rescaled = miu.rescale_sample(sample_input_MC[:, 2])
    gamma_bar_fs_list_rescaled = miu.rescale_sample(sample_input_MC[:, 3])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(sample_input_MC[:, 4])
    r_bar_list_rescaled = miu.rescale_sample(sample_input_MC[:, 5])
    input_sample_rescaled = ot.Sample(len(gamma_bar_0_list_rescaled), 6)
    for k in range(len(gamma_bar_0_list_rescaled)):
        input_sample_rescaled[k, 0] = gamma_bar_0_list_rescaled[k]
        input_sample_rescaled[k, 1] = sigma_bar_list_rescaled[k]
        input_sample_rescaled[k, 2] = gamma_bar_r_list_rescaled[k]
        input_sample_rescaled[k, 3] = gamma_bar_fs_list_rescaled[k]
        input_sample_rescaled[k, 4] = gamma_bar_lambda_list_rescaled[k]
        input_sample_rescaled[k, 5] = r_bar_list_rescaled[k]

    input_sample = sample_input_MC  # ot.Sample(shuffled_sample[:, 0:6])
    output_model = shuffled_sample[:, -2]
    output_pce = metamodel_pce(input_sample_rescaled)
    output_kriging = metamodel_kriging(input_sample)

    X_plot = np.linspace(0, 1, 2000)[:, None]

    kde_model = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_model)
    log_dens_model = kde_model.score_samples(X_plot)
    kde_kriging = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_kriging)
    log_dens_kriging = kde_kriging.score_samples(X_plot)
    kde_pce = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_pce)
    log_dens_pce = kde_pce.score_samples(X_plot)

    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    ax.hist(output_model, bins=20, density=True, color="lightgray", alpha=0.3, ec="black")

    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_model),
        color="k",
        alpha=0.3,
        lw=4,
        linestyle="-",
        label="model",
    )
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_kriging),
        color=purple,
        lw=4,
        linestyle="--",
        label="Kriging",
    )
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_pce),
        color=orange,
        lw=4,
        linestyle=":",
        label="PCE",
    )
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(
        [0, 0.2, 0.4, 0.6, 0.8, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 2, 4, 6])
    ax.set_yticklabels(
        [0, 2, 4, 6],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-0.02, 1.02))
    ax.set_ylim((0, 7))
    ax.set_xlabel(r"$\tilde{f}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$p_{\tilde{F}}(\tilde{f})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax.grid(linestyle="--")
    savefigure.save_as_png(fig, "PDF_metamodel_mechanoadaptation_vs_passive_elliptic" + str(pixels))


def plot_PDF_pce_kriging_mechanoadaptation_vs_passive_circular(metamodelposttreatment, degree, training_amount):
    """Plots the Probability Density Functions (PDFs) of the metamodel outputs and original data

    Parameters:
    ----------
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    degree: int
        Dimension of the basis of polynoms
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    distribution_input = ot.ComposedDistribution(
        [ot.Uniform(1, 8), ot.Uniform(0.5, 5.5), ot.Uniform(1, 6), ot.Uniform(-0.45, 0.45), ot.Uniform(10, 100)]
    )
    experiment_input = ot.MonteCarloExperiment(distribution_input, int(1e5))
    sample_input_MC = experiment_input.generate()
    complete_pkl_filename_pce = miu.create_pkl_name(
        "PCE_mechanoadaptation_vs_passive_circular" + str(degree), training_amount
    )
    shuffled_sample, results_from_algo_pce = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_pce)
    metamodel_pce = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_pce)
    complete_pkl_filename_kriging = miu.create_pkl_name(
        "Kriging_mechanoadaptation_vs_passive_circular", training_amount
    )
    _, results_from_algo_kriging = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_kriging)
    metamodel_kriging = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_kriging)

    gamma_bar_0_list_rescaled = miu.rescale_sample(sample_input_MC[:, 0])
    sigma_bar_list_rescaled = miu.rescale_sample(sample_input_MC[:, 1])
    gamma_bar_r_list_rescaled = miu.rescale_sample(sample_input_MC[:, 2])
    gamma_bar_fs_bar_list_rescaled = miu.rescale_sample(sample_input_MC[:, 3])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(sample_input_MC[:, 4])
    input_sample_rescaled = ot.Sample(len(gamma_bar_r_list_rescaled), 5)
    for k in range(len(gamma_bar_r_list_rescaled)):
        input_sample_rescaled[k, 0] = gamma_bar_0_list_rescaled[k]
        input_sample_rescaled[k, 1] = sigma_bar_list_rescaled[k]
        input_sample_rescaled[k, 2] = gamma_bar_r_list_rescaled[k]
        input_sample_rescaled[k, 3] = gamma_bar_fs_bar_list_rescaled[k]
        input_sample_rescaled[k, 4] = gamma_bar_lambda_list_rescaled[k]

    input_sample = sample_input_MC  # ot.Sample(shuffled_sample[:, 0:5])
    output_model = shuffled_sample[:, -2]
    output_pce = metamodel_pce(input_sample_rescaled)
    output_kriging = metamodel_kriging(input_sample)

    X_plot = np.linspace(0, 1, 1000)[:, None]

    kde_model = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_model)
    log_dens_model = kde_model.score_samples(X_plot)
    kde_kriging = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_kriging)
    log_dens_kriging = kde_kriging.score_samples(X_plot)
    kde_pce = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_pce)
    log_dens_pce = kde_pce.score_samples(X_plot)

    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    ax.hist(shuffled_sample[:, -2], bins=20, density=True, color="lightgray", alpha=0.3, ec="black")
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_model),
        color="k",
        alpha=0.4,
        lw=4,
        linestyle="-",
        label="model",
    )
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_kriging),
        color=purple,
        lw=4,
        linestyle=":",
        label="Kriging",
    )
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_pce),
        color=orange,
        lw=4,
        linestyle="--",
        label="PCE",
    )
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(
        [0, 0.2, 0.4, 0.6, 0.8, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 2, 4, 6])
    ax.set_yticklabels(
        [0, 2, 4, 6],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-0.02, 1.02))
    ax.set_xlabel(r"$\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$p_{\tilde{F}}(\tilde{f})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    savefigure.save_as_png(fig, "PDF_metamodel_circular_mechanoadaptation_vs_passive_circular" + str(pixels))


if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    xticks = XTicks()
    xticklabels = XTickLabels()
    pixels = 360

    metamodelposttreatment = MetamodelPostTreatment()
    metamodelvalidation = MetamodelValidation()

    ### Constant elliptic ###

    # filename_qMC_constant_elliptic = "dataset_for_metamodel_creation_feq_constant_elliptic.txt"
    # training_amount_constant_elliptic = 0.9
    # degree_list_constant_elliptic = np.arange(1, 21)

    # datapresetting_constant_elliptic = DataPreSetting(filename_qMC_constant_elliptic, training_amount_constant_elliptic)
    # optimal_degree_constant_elliptic = optimize_degree_pce_constant_elliptic(
    #                                                                          datapresetting_constant_elliptic,
    #                                                                          metamodelposttreatment,
    #                                                                          metamodelvalidation,
    #                                                                          degree_list_constant_elliptic,
    #                                                                          training_amount_constant_elliptic,
    #                                                                          createfigure,
    #                                                                          savefigure,
    #                                                                          xticks,
    #                                                                          pixels,
    #                                                                     )
    # metamodel_validation_routine_pce_constant_elliptic(
    #     datapresetting_constant_elliptic,
    #     metamodelposttreatment,
    #     metamodelvalidation,
    #     'PCE',
    #     training_amount_constant_elliptic,
    #     optimal_degree_constant_elliptic,
    #     createfigure,
    #     savefigure,
    #     xticks,
    #     pixels,
    # )

    # metamodel_validation_routine_kriging_constant_elliptic(
    #     datapresetting_constant_elliptic,
    #     metamodelposttreatment,
    #     metamodelvalidation,
    #     'Kriging',
    #     training_amount_constant_elliptic,
    #     createfigure,
    #     savefigure,
    #     xticks,
    #     pixels,
    # )

    # plot_PDF_pce_kriging_constant_elliptic(metamodelposttreatment, optimal_degree_constant_elliptic, training_amount_constant_elliptic)

    ## Mechanoadaptation circular ###

    # filename_qMC_mechanoadaptation_circular = "dataset_for_metamodel_creation_mechanoadaptation_circular.txt"
    # training_amount_mechanoadaptation_circular = 0.8
    # degree_list_mechanoadaptation_circular = np.arange(1, 12)

    # datapresetting_mechanoadaptation_circular = DataPreSetting(
    #     filename_qMC_mechanoadaptation_circular, training_amount_mechanoadaptation_circular
    # )
    # optimal_degree_mechanoadaptation_circular = optimize_degree_pce_mechanoadaptation_circular(
    #                                                                          datapresetting_mechanoadaptation_circular,
    #                                                                          metamodelposttreatment,
    #                                                                          metamodelvalidation,
    #                                                                          degree_list_mechanoadaptation_circular,
    #                                                                          training_amount_mechanoadaptation_circular,
    #                                                                          createfigure,
    #                                                                          savefigure,
    #                                                                          xticks,
    #                                                                          pixels,
    #                                                                     )
    # metamodel_validation_routine_pce_mechanoadaptation_circular(
    #     datapresetting_mechanoadaptation_circular,
    #     metamodelposttreatment,
    #     metamodelvalidation,
    #     'PCE',
    #     training_amount_mechanoadaptation_circular,
    #     optimal_degree_mechanoadaptation_circular,
    #     createfigure,
    #     savefigure,
    #     xticks,
    #     pixels,
    # )

    # metamodel_validation_routine_kriging_mechanoadaptation_circular(
    #     datapresetting_mechanoadaptation_circular,
    #     metamodelposttreatment,
    #     metamodelvalidation,
    #     'Kriging',
    #     training_amount_mechanoadaptation_circular,
    #     createfigure,
    #     savefigure,
    #     xticks,
    #     pixels,
    # )

    # plot_PDF_pce_kriging_mechanoadaptation_circular(metamodelposttreatment, optimal_degree_mechanoadaptation_circular, training_amount_mechanoadaptation_circular)
    # plot_PDF_pce_kriging_mechanoadaptation_circular(
    #     metamodelposttreatment, 1, training_amount_mechanoadaptation_circular
    # )

    ### Mechanoadaptation vs passive circular ###

    # filename_qMC_mechanoadaptation_vs_passive_circular = "dataset_for_metamodel_creation_mechanoadaptation_vs_passive_circular.txt"
    # training_amount_mechanoadaptation_vs_passive_circular = 0.8
    # degree_list_mechanoadaptation_vs_passive_circular = np.arange(1, 8)

    # datapresetting_mechanoadaptation_vs_passive_circular = DataPreSetting(filename_qMC_mechanoadaptation_vs_passive_circular, training_amount_mechanoadaptation_vs_passive_circular)
    # optimal_degree_mechanoadaptation_vs_passive_circular = optimize_degree_pce_mechanoadaptation_vs_passive_circular(
    #                                                                          datapresetting_mechanoadaptation_vs_passive_circular,
    #                                                                          metamodelposttreatment,
    #                                                                          metamodelvalidation,
    #                                                                          degree_list_mechanoadaptation_vs_passive_circular,
    #                                                                          training_amount_mechanoadaptation_vs_passive_circular,
    #                                                                          createfigure,
    #                                                                          savefigure,
    #                                                                          xticks,
    #                                                                          pixels,
    #                                                                     )
    # metamodel_validation_routine_pce_mechanoadaptation_vs_passive_circular(
    #     datapresetting_mechanoadaptation_vs_passive_circular,
    #     metamodelposttreatment,
    #     metamodelvalidation,
    #     'PCE',
    #     training_amount_mechanoadaptation_vs_passive_circular,
    #     optimal_degree_mechanoadaptation_vs_passive_circular,
    #     createfigure,
    #     savefigure,
    #     xticks,
    #     pixels,
    # )

    # metamodel_validation_routine_kriging_mechanoadaptation_vs_passive_circular(
    #     datapresetting_mechanoadaptation_vs_passive_circular,
    #     metamodelposttreatment,
    #     metamodelvalidation,
    #     'Kriging',
    #     training_amount_mechanoadaptation_vs_passive_circular,
    #     createfigure,
    #     savefigure,
    #     xticks,
    #     pixels,
    # )

    # plot_PDF_pce_kriging_mechanoadaptation_vs_passive_circular(metamodelposttreatment, optimal_degree_mechanoadaptation_vs_passive_circular, training_amount_mechanoadaptation_vs_passive_circular)

    ## Mechanoadaptation elliptic ###

    filename_qMC_mechanoadaptation_elliptic = "dataset_for_metamodel_creation_mechanoadaptation_elliptic.txt"
    training_amount_mechanoadaptation_elliptic = 0.65
    degree_list_mechanoadaptation_elliptic = np.arange(1, 7)

    # datapresetting_mechanoadaptation_elliptic = DataPreSetting(filename_qMC_mechanoadaptation_elliptic, training_amount_mechanoadaptation_elliptic)
    # optimal_degree_mechanoadaptation_elliptic = optimize_degree_pce_mechanoadaptation_elliptic(
    #                                                                          datapresetting_mechanoadaptation_elliptic,
    #                                                                          metamodelposttreatment,
    #                                                                          metamodelvalidation,
    #                                                                          degree_list_mechanoadaptation_elliptic,
    #                                                                          training_amount_mechanoadaptation_elliptic,
    #                                                                          createfigure,
    #                                                                          savefigure,
    #                                                                          xticks,
    #                                                                          pixels,
    #                                                                     )
    # metamodel_validation_routine_pce_mechanoadaptation_elliptic(
    #     datapresetting_mechanoadaptation_elliptic,
    #     metamodelposttreatment,
    #     metamodelvalidation,
    #     'PCE',
    #     training_amount_mechanoadaptation_elliptic,
    #     6,#optimal_degree_mechanoadaptation_elliptic,
    #     createfigure,
    #     savefigure,
    #     xticks,
    #     pixels,
    # )

    # metamodel_validation_routine_kriging_mechanoadaptation_elliptic(
    #     datapresetting_mechanoadaptation_elliptic,
    #     metamodelposttreatment,
    #     metamodelvalidation,
    #     'Kriging',
    #     training_amount_mechanoadaptation_elliptic,
    #     createfigure,
    #     savefigure,
    #     xticks,
    #     pixels,
    # )

    plot_PDF_pce_kriging_mechanoadaptation_elliptic(
        metamodelposttreatment, 1, training_amount_mechanoadaptation_elliptic
    )
    # plot_PDF_pce_kriging_mechanoadaptation_elliptic(metamodelposttreatment, optimal_degree_mechanoadaptation_elliptic, training_amount_mechanoadaptation_elliptic)

    ## Mechanoadaptation vs passive elliptic ###

    # filename_qMC_mechanoadaptation_vs_passive_elliptic = "dataset_for_metamodel_creation_mechanoadaptation_vs_passive_elliptic.txt"
    # training_amount_mechanoadaptation_vs_passive_elliptic = 0.9
    # degree_list_mechanoadaptation_vs_passive_elliptic = np.arange(1, 6)

    # datapresetting_mechanoadaptation_vs_passive_elliptic = DataPreSetting(filename_qMC_mechanoadaptation_vs_passive_elliptic, training_amount_mechanoadaptation_vs_passive_elliptic)
    # optimal_degree_mechanoadaptation_vs_passive_elliptic = optimize_degree_pce_mechanoadaptation_vs_passive_elliptic(
    #                                                                          datapresetting_mechanoadaptation_vs_passive_elliptic,
    #                                                                          metamodelposttreatment,
    #                                                                          metamodelvalidation,
    #                                                                          degree_list_mechanoadaptation_vs_passive_elliptic,
    #                                                                          training_amount_mechanoadaptation_vs_passive_elliptic,
    #                                                                          createfigure,
    #                                                                          savefigure,
    #                                                                          xticks,
    #                                                                          pixels,
    #                                                                     )
    # metamodel_validation_routine_pce_mechanoadaptation_vs_passive_elliptic(
    #     datapresetting_mechanoadaptation_vs_passive_elliptic,
    #     metamodelposttreatment,
    #     metamodelvalidation,
    #     'PCE',
    #     training_amount_mechanoadaptation_vs_passive_elliptic,
    #     optimal_degree_mechanoadaptation_vs_passive_elliptic,
    #     createfigure,
    #     savefigure,
    #     xticks,
    #     pixels,
    # )

    # metamodel_validation_routine_kriging_mechanoadaptation_vs_passive_elliptic(
    #     datapresetting_mechanoadaptation_vs_passive_elliptic,
    #     metamodelposttreatment,
    #     metamodelvalidation,
    #     'Kriging',
    #     training_amount_mechanoadaptation_vs_passive_elliptic,
    #     createfigure,
    #     savefigure,
    #     xticks,
    #     pixels,
    # )

    # plot_PDF_pce_kriging_mechanoadaptation_vs_passive_elliptic(metamodelposttreatment, optimal_degree_mechanoadaptation_vs_passive_elliptic, training_amount_mechanoadaptation_vs_passive_elliptic)
