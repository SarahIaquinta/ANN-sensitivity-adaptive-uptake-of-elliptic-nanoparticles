import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import seaborn as sns

ot.Log.Show(ot.Log.NONE)
import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_evaluation import plot

import uptake.metamodel_implementation
import uptake.metamodel_implementation.ANN.utils as ANN_miu
import uptake.metamodel_implementation.utils as miu
from uptake.figures.utils import CreateFigure, Fonts, SaveFigure, XTickLabels, XTicks
from uptake.metamodel_implementation.metamodel_creation import DataPreSetting


def test_sklearn_constant_elliptic(X_train, X_test, y_train, y_test):
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)

    reg = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation="relu", random_state=1, max_iter=2000).fit(
        X_trainscaled, y_train
    )

    y_pred = reg.predict(X_testscaled)
    Q2 = r2_score(y_pred, y_test)
    print("The Score with ", Q2)
    ANN_miu.export_predictions(y_test, y_pred, Q2, "predictions_ANN_constant_elliptic.pkl")


def test_sklearn_mechanoadaptation_vs_passive_elliptic(X_train, X_test, y_train, y_test, complete_filename_ANN):
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)
    reg = MLPRegressor(
        hidden_layer_sizes=(64, 64, 64, 1), activation="relu", solver="lbfgs", random_state=1, max_iter=500
    ).fit(X_trainscaled, y_train)
    y_pred = reg.predict(X_testscaled)
    Q2 = r2_score(y_pred, y_test)
    print("The Score with ", Q2)
    ANN_miu.export_predictions(y_test, y_pred, Q2, complete_filename_ANN)


def test_sklearn_mechanoadaptation_vs_passive_elliptic_new_settings(
    X_train, X_test, y_train, y_test, complete_filename_ANN
):
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)
    reg = MLPRegressor(
        hidden_layer_sizes=(64, 64, 64, 1), activation="relu", solver="lbfgs", random_state=1, max_iter=500
    ).fit(X_trainscaled, y_train)
    y_pred = reg.predict(X_testscaled)
    Q2 = r2_score(y_pred, y_test)
    print("The Score with ", Q2)
    ANN_miu.export_predictions(y_test, y_pred, Q2, complete_filename_ANN)


def ANN_gridsearch_mechanoadaptation_vs_passive_elliptic(X_train, X_test, y_train, y_test, complete_filename_ANN):
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)
    mlpr = MLPRegressor(max_iter=7000, early_stopping=True)
    net1 = (
        64,
        64,
        64,
        1,
    )
    net2 = (
        64,
        64,
        64,
        2,
    )
    net3 = (
        64,
        64,
        64,
        64,
        1,
    )
    net4 = (
        64,
        64,
        64,
        8,
        1,
    )
    net5 = (64, 32, 16, 8, 4, 2, 1)
    net6 = (128, 128, 128, 1)
    param_list = {
        "hidden_layer_sizes": [net3],
        "activation": ["tanh"],
        "solver": ["lbfgs"],
        "learning_rate": ["adaptive"],
    }
    gridCV = GridSearchCV(estimator=mlpr, param_grid=param_list, n_jobs=os.cpu_count() - 2)
    complete_filename_grid = "grid_search.pkl"
    y_train = np.ravel(y_train)
    gridCV.fit(X_trainscaled, y_train)
    y_test = np.ravel(y_test)
    ANN_miu.export_gridsearch(sc_X, gridCV, complete_filename_grid)
    results = sorted(gridCV.cv_results_.keys())
    y_pred = gridCV.predict(X_testscaled)
    Q2 = r2_score(y_pred, y_test)
    print("The Score with ", Q2)
    ANN_miu.export_predictions(y_test, y_pred, Q2, complete_filename_ANN)


def ANN_gridsearch_mechanoadaptation_vs_passive_elliptic_new_settings(
    X_train, X_test, y_train, y_test, complete_filename_ANN
):
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)
    mlpr = MLPRegressor(max_iter=7000, early_stopping=True)
    net1 = (
        8,
    )
    net2 = (
8,4, 2,
    )
    net3 = (
        16, 8, 4, 2
    )
    net4 = (8,8,8)
    # net5 = (16,16,16)

    net5 = (64, 32, 16, 8, 4, 2, 1)
    net6 = (64, 32, 16, 8, 4, 2)
    # param_list = {
    #     "hidden_layer_sizes": [net1, net2, net3, net4, net5, net6],
    #     "activation": ["relu","tanh"],
    #     "solver": ["adam", "lbfgs"],
    #     "learning_rate": ["adaptive"],
    # }
    param_list = {
        "hidden_layer_sizes": [ net4],
        "activation": ["tanh"],
        "solver": ["adam", "lbfgs"],
        "learning_rate": ["adaptive"],
    }
    gridCV = GridSearchCV(estimator=mlpr, param_grid=param_list, n_jobs=os.cpu_count() - 3, refit=True)
    complete_filename_grid_new_settings = "grid_search_article-v2.pkl"
    y_train = np.ravel(y_train)
    gridCV.fit(X_trainscaled, y_train)
    y_test = np.ravel(y_test)
    ANN_miu.export_gridsearch(sc_X, gridCV, complete_filename_grid_new_settings)
    results = sorted(gridCV.cv_results_.keys())
    y_pred = gridCV.predict(X_testscaled)
    Q2 = r2_score(y_pred, y_test)
    print("Q2 ANN : ", Q2)
    ANN_miu.export_predictions(y_test, y_pred, Q2, complete_filename_ANN)


def observe_results_gridsearch_learning_rate(complete_filename_grid, fonts, createfigure, savefigure):
    _, grid = ANN_miu.extract_gridsearch(complete_filename_grid)
    results = grid.cv_results_
    fig = createfigure.rectangle_figure(pixels=pixels)
    ax = fig.gca()
    ax = plot.grid_search(results, change="learning_rate", kind="bar")
    ax.set_xlabel("Learning rate", font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_ylabel(r"$Q_2^{ANN}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], font=fonts.serif(), fontsize=fonts.axis_legend_size())

    ax.set_xticklabels(["constant", "adaptive"], font=fonts.serif(), fontsize=fonts.axis_legend_size())
    legend = ax.legend(prop=fonts.serif_gridsearch(), framealpha=0.7)
    legend.remove()
    ax.set_title("")
    savefigure.save_as_png(fig, "gridsearch_ANN_learning_rate")


def observe_results_gridsearch_solver(complete_filename_grid, fonts, createfigure, savefigure):
    _, grid = ANN_miu.extract_gridsearch(complete_filename_grid)
    results = grid.cv_results_
    fig = createfigure.rectangle_figure(pixels=pixels)
    ax = fig.gca()
    ax = plot.grid_search(results, change="solver", subset={"learning_rate": "adaptive"}, kind="bar")
    ax.set_xlabel("Solver", font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_ylabel(r"$Q_2^{ANN}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_xticklabels(["lbfgs", "sgd", "adam"], font=fonts.serif(), fontsize=fonts.axis_legend_size())
    legend = ax.legend(prop=fonts.serif_gridsearch(), framealpha=0.7)
    legend.remove()
    ax.set_title("")
    savefigure.save_as_png(fig, "gridsearch_ANN_solver")


def observe_results_gridsearch_activation(complete_filename_grid, fonts, createfigure, savefigure):
    _, grid = ANN_miu.extract_gridsearch(complete_filename_grid)
    results = grid.cv_results_
    fig = createfigure.rectangle_figure(pixels=pixels)
    ax = fig.gca()
    ax = plot.grid_search(
        results, change="activation", subset={"learning_rate": "adaptive", "solver": "adam"}, kind="bar"
    )
    ax.set_xlabel("Activation function", font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_ylabel(r"$Q_2^{ANN}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_xticklabels(["ReLU", "tanh"], font=fonts.serif(), fontsize=fonts.axis_legend_size())
    legend = ax.legend(prop=fonts.serif_gridsearch(), framealpha=0.7)
    legend.remove()
    ax.set_title("")
    savefigure.save_as_png(fig,"gridsearch_ANN_activation_article-v3")
    savefigure.save_as_svg(fig, "gridsearch_ANN_activation_article-v3")


def observe_results_gridsearch_architecture(complete_filename_grid, fonts, createfigure, savefigure):
    _, grid = ANN_miu.extract_gridsearch(complete_filename_grid)
    results = grid.cv_results_
    fig = createfigure.rectangle_figure(pixels=pixels)
    ax = fig.gca()
    ax = plot.grid_search(
        results,
        change="hidden_layer_sizes",
        subset={"learning_rate": "adaptive", "solver": "adam", "activation": "tanh"},
        kind="bar",
    )
    ax.set_xlabel("Architecture", font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_ylabel(r"$Q_2^{ANN}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(
        [
            "(a)",
            "(b)",
            "(c)",
            "(d)",
            "(e)",
            "(f)",
        ],
        font=fonts.serif(), fontsize=fonts.axis_legend_size()
    )
    legend = ax.legend(prop=fonts.serif_gridsearch(), framealpha=0.7)
    legend.remove()
    ax.set_title("")
    savefigure.save_as_png(fig, "gridsearch_ANN_architecture-v2")


def plot_true_vs_predicted_constant_elliptic(y_test, y_pred, createfigure, savefigure, fonts, pixels):
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    color_plot = orange
    ax.plot(y_test, y_pred, "o", color=color_plot)
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
    savefigure.save_as_png(fig, "ANN_constant_elliptic_" + str(pixels))


def plot_true_vs_predicted_mechanoadaptation_vs_passive_elliptic(
    y_test, y_pred, createfigure, savefigure, fonts, pixels
):
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    color_plot = orange
    ax.plot(y_test, y_pred, "o", color=color_plot)
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
    savefigure.save_as_png(fig, "ANN_mechanoadaptation_vs_passive_elliptic_article_" + str(pixels))


def plot_true_vs_predicted_mechanoadaptation_vs_passive_elliptic_new_settings(
    y_test, y_pred, createfigure, savefigure, fonts, pixels
):
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    color_plot = orange
    ax.plot(y_test, y_pred, "o", color=color_plot)
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
    savefigure.save_as_png(fig, "ANN_mechanoadaptation_vs_passive_elliptic_new_settings_article-v2_" + str(pixels))


def routine_constant_elliptic(fonts, createfigure, savefigure, pixels):
    filename_qMC_constant_elliptic = "dataset_for_metamodel_creation_feq_constant_elliptic.txt"
    training_amount_constant_elliptic = 0.9
    datapresetting_constant_elliptic = DataPreSetting(filename_qMC_constant_elliptic, training_amount_constant_elliptic)
    shuffled_sample = datapresetting_constant_elliptic.shuffle_dataset_from_datafile()
    (
        X_train_constant_elliptic,
        y_train_constant_elliptic,
    ) = datapresetting_constant_elliptic.extract_training_data_from_shuffled_dataset_constant_elliptic(shuffled_sample)
    (
        X_test_constant_elliptic,
        y_test_constant_elliptic,
    ) = datapresetting_constant_elliptic.extract_testing_data_constant_elliptic(shuffled_sample)
    test_sklearn_constant_elliptic(
        X_train_constant_elliptic, X_test_constant_elliptic, y_train_constant_elliptic, y_test_constant_elliptic
    )
    y_test_constant_elliptic, y_pred_constant_elliptic, Q2_constant_elliptic = ANN_miu.extract_predictions(
        "predictions_ANN_constant_elliptic.pkl"
    )
    plot_true_vs_predicted_constant_elliptic(
        y_test_constant_elliptic, y_pred_constant_elliptic, createfigure, savefigure, fonts, pixels
    )


# def routine_mechanoadaptation_vs_passive_elliptic(fonts, createfigure, savefigure, pixels):
#     filename_qMC_mechanoadaptation_vs_passive_elliptic = (
#         "dataset_for_metamodel_creation_mechanoadaptation_vs_passive_elliptic.txt"
#     )
#     training_amount_mechanoadaptation_vs_passive_elliptic = 0.9
#     datapresetting_mechanoadaptation_vs_passive_elliptic = DataPreSetting(
#         filename_qMC_mechanoadaptation_vs_passive_elliptic, training_amount_mechanoadaptation_vs_passive_elliptic
#     )
#     shuffled_sample = datapresetting_mechanoadaptation_vs_passive_elliptic.shuffle_dataset_from_datafile()
#     (
#         X_train_mechanoadaptation_vs_passive_elliptic,
#         y_train_mechanoadaptation_vs_passive_elliptic,
#     ) = datapresetting_mechanoadaptation_vs_passive_elliptic.extract_training_data_from_shuffled_dataset_mechanoadaptation_vs_passive_elliptic(
#         shuffled_sample
#     )
#     (
#         X_test_mechanoadaptation_vs_passive_elliptic,
#         y_test_mechanoadaptation_vs_passive_elliptic,
#     ) = datapresetting_mechanoadaptation_vs_passive_elliptic.extract_testing_data_mechanoadaptation_vs_passive_elliptic(
#         shuffled_sample
#     )
#     test_sklearn_mechanoadaptation_vs_passive_elliptic(
#         X_train_mechanoadaptation_vs_passive_elliptic,
#         X_test_mechanoadaptation_vs_passive_elliptic,
#         y_train_mechanoadaptation_vs_passive_elliptic,
#         y_test_mechanoadaptation_vs_passive_elliptic,
#     )
#     (
#         y_test_mechanoadaptation_vs_passive_elliptic,
#         y_pred_mechanoadaptation_vs_passive_elliptic,
#         Q2_mechanoadaptation_vs_passive_elliptic,
#     ) = ANN_miu.extract_predictions("predictions_ANN_mechanoadaptation_vs_passive_elliptic.pkl")
#     plot_true_vs_predicted_mechanoadaptation_vs_passive_elliptic(
#         y_test_mechanoadaptation_vs_passive_elliptic,
#         y_pred_mechanoadaptation_vs_passive_elliptic,
#         createfigure,
#         savefigure,
#         fonts,
#         pixels,
#     )


def routine_mechanoadaptation_vs_passive_elliptic_new_settings(fonts, createfigure, savefigure, pixels):
    filename_qMC_mechanoadaptation_vs_passive_elliptic_new_settings = (
        "dataset_for_ANN_mechanadaptation_vs_passive_elliptic_newsettings_size4.txt"
    )
    training_amount_mechanoadaptation_vs_passive_elliptic = 0.9
    datapresetting_mechanoadaptation_vs_passive_elliptic_new_settings = DataPreSetting(
        filename_qMC_mechanoadaptation_vs_passive_elliptic_new_settings,
        training_amount_mechanoadaptation_vs_passive_elliptic,
    )
    shuffled_sample = datapresetting_mechanoadaptation_vs_passive_elliptic_new_settings.shuffle_dataset_from_datafile()
    (
        X_train_mechanoadaptation_vs_passive_elliptic,
        y_train_mechanoadaptation_vs_passive_elliptic,
    ) = datapresetting_mechanoadaptation_vs_passive_elliptic_new_settings.extract_training_data_from_shuffled_dataset_mechanoadaptation_vs_passive_elliptic(
        shuffled_sample
    )
    (
        X_test_mechanoadaptation_vs_passive_elliptic,
        y_test_mechanoadaptation_vs_passive_elliptic,
    ) = datapresetting_mechanoadaptation_vs_passive_elliptic_new_settings.extract_testing_data_mechanoadaptation_vs_passive_elliptic(
        shuffled_sample
    )
    complete_filename_ANN = "predictions_ANN_mechanoadaptation_vs_passive_elliptic_new_settings-v2.pkl"
    
    ANN_gridsearch_mechanoadaptation_vs_passive_elliptic_new_settings(
        X_train_mechanoadaptation_vs_passive_elliptic,
        X_test_mechanoadaptation_vs_passive_elliptic,
        y_train_mechanoadaptation_vs_passive_elliptic,
        y_test_mechanoadaptation_vs_passive_elliptic,
        complete_filename_ANN,
    )
    complete_filename_grid = "grid_search_article-v2.pkl"

    scaler, grid = ANN_miu.extract_gridsearch(complete_filename_grid)
    scaled_x_test = scaler.transform(X_test_mechanoadaptation_vs_passive_elliptic)
    y_pred = grid.predict(scaled_x_test)
    plt.figure()
    plt.plot(y_test_mechanoadaptation_vs_passive_elliptic, y_pred)
    Q2 = r2_score(y_pred, y_test_mechanoadaptation_vs_passive_elliptic)
    print("Q2 ANN : ", Q2)
    

    (
        y_test_mechanoadaptation_vs_passive_elliptic,
        y_pred_mechanoadaptation_vs_passive_elliptic,
        Q2_mechanoadaptation_vs_passive_elliptic,
    ) = ANN_miu.extract_predictions(complete_filename_ANN)
    plot_true_vs_predicted_mechanoadaptation_vs_passive_elliptic_new_settings(
        y_test_mechanoadaptation_vs_passive_elliptic,
        y_pred_mechanoadaptation_vs_passive_elliptic,
        createfigure,
        savefigure,
        fonts,
        pixels,
    )


def routine_mechanoadaptation_vs_passive_elliptic(fonts, createfigure, savefigure, pixels):
    filename_qMC_mechanoadaptation_vs_passive_elliptic = "dataset_for_ANN_mechanadaptation_vs_passive_elliptic.txt"
    training_amount_mechanoadaptation_vs_passive_elliptic = 0.9
    datapresetting_mechanoadaptation_vs_passive_elliptic = DataPreSetting(
        filename_qMC_mechanoadaptation_vs_passive_elliptic,
        training_amount_mechanoadaptation_vs_passive_elliptic,
    )
    shuffled_sample = datapresetting_mechanoadaptation_vs_passive_elliptic.shuffle_dataset_from_datafile()
    (
        X_train_mechanoadaptation_vs_passive_elliptic,
        y_train_mechanoadaptation_vs_passive_elliptic,
    ) = datapresetting_mechanoadaptation_vs_passive_elliptic.extract_training_data_from_shuffled_dataset_mechanoadaptation_vs_passive_elliptic(
        shuffled_sample
    )
    (
        X_test_mechanoadaptation_vs_passive_elliptic,
        y_test_mechanoadaptation_vs_passive_elliptic,
    ) = datapresetting_mechanoadaptation_vs_passive_elliptic.extract_testing_data_mechanoadaptation_vs_passive_elliptic(
        shuffled_sample
    )
    complete_filename_ANN = "predictions_ANN_mechanoadaptation_vs_passive_elliptic.pkl"
    ANN_gridsearch_mechanoadaptation_vs_passive_elliptic(
        X_train_mechanoadaptation_vs_passive_elliptic,
        X_test_mechanoadaptation_vs_passive_elliptic,
        y_train_mechanoadaptation_vs_passive_elliptic,
        y_test_mechanoadaptation_vs_passive_elliptic,
        complete_filename_ANN,
    )
    (
        y_test_mechanoadaptation_vs_passive_elliptic,
        y_pred_mechanoadaptation_vs_passive_elliptic,
        Q2_mechanoadaptation_vs_passive_elliptic,
    ) = ANN_miu.extract_predictions(complete_filename_ANN)
    plot_true_vs_predicted_mechanoadaptation_vs_passive_elliptic(
        y_test_mechanoadaptation_vs_passive_elliptic,
        y_pred_mechanoadaptation_vs_passive_elliptic,
        createfigure,
        savefigure,
        fonts,
        pixels,
    )


if __name__ == "__main__":
    fonts = Fonts()
    createfigure = CreateFigure()
    savefigure = SaveFigure()
    pixels = 360

    # routine_constant_elliptic(fonts, createfigure, savefigure, pixels)

    # routine_mechanoadaptation_vs_passive_elliptic(fonts, createfigure, savefigure, pixels)

    routine_mechanoadaptation_vs_passive_elliptic_new_settings(fonts, createfigure, savefigure, pixels)

    complete_filename_grid = "grid_search_article-v2.pkl"
    # observe_results_gridsearch_learning_rate(complete_filename_grid, fonts, createfigure, savefigure)
    # observe_results_gridsearch_solver(complete_filename_grid, fonts, createfigure, savefigure)
    observe_results_gridsearch_activation(complete_filename_grid, fonts, createfigure, savefigure)
    # observe_results_gridsearch_architecture(complete_filename_grid, fonts, createfigure, savefigure)
    # complete_filename_ANN = "predictions_ANN_mechanoadaptation_vs_passive_elliptic_new_settings-v2.pkl"
    # routine_mechanoadaptation_vs_passive_elliptic_new_settings(fonts, createfigure, savefigure, pixels)
    # _, grid = ANN_miu.extract_gridsearch(complete_filename_grid)
    # results = grid.cv_results_
    # y_pred = gridCV.predict(X_testscaled)
    # Q2 = r2_score(y_pred, y_test)
    # print("Q2 ANN : ", Q2)
    # ANN_miu.export_predictions(y_test, y_pred, Q2, complete_filename_ANN)
    
    # (
    #     y_test_mechanoadaptation_vs_passive_elliptic,
    #     y_pred_mechanoadaptation_vs_passive_elliptic,
    #     Q2_mechanoadaptation_vs_passive_elliptic,
    # ) = ANN_miu.extract_predictions(complete_filename_ANN)
    # # _, grid = ANN_miu.extract_gridsearch(complete_filename_grid)
    

