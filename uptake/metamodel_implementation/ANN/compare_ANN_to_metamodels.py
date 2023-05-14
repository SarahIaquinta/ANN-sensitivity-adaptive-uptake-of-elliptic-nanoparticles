import time
from pathlib import Path

import numpy as np
import openturns as ot
import seaborn as sns

import uptake
import uptake.metamodel_implementation.ANN.implementation as imp_ANN
import uptake.metamodel_implementation.ANN.utils as ANN_miu
import uptake.metamodel_implementation.utils as miu
from uptake.figures.utils import CreateFigure, Fonts, SaveFigure, XTickLabels, XTicks
from uptake.metamodel_implementation.metamodel_creation import DataPreSetting, MetamodelCreation
from uptake.metamodel_implementation.metamodel_validation import MetamodelPostTreatment, MetamodelValidation

ot.Log.Show(ot.Log.NONE)
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler


def build_predictions_ANN(X_train, X_test, y_train, y_test):
    complete_filename_ANN = "predictions_ANN_mechanoadaptation_vs_passive_elliptic.pkl"
    complete_filename_ANN_size16 = "predictions_ANN_size16_mechanoadaptation_vs_passive_elliptic.pkl"
    # imp_ANN.test_sklearn_mechanoadaptation_vs_passive_elliptic(X_train, X_test, y_train, y_test, complete_filename_ANN)
    imp_ANN.ANN_gridsearch_mechanoadaptation_vs_passive_elliptic(
        X_train, X_test, y_train, y_test, complete_filename_ANN_size16
    )


def build_predictions_ANN_new_settings(X_train, X_test, y_train, y_test):
    complete_filename_ANN = "predictions_ANN_mechanoadaptation_vs_passive_elliptic.pkl"
    complete_filename_ANN_size4_article = "predictions_ANN_mechanoadaptation_vs_passive_elliptic_new_settings-v2.pkl"
    # imp_ANN.test_sklearn_mechanoadaptation_vs_passive_elliptic(X_train, X_test, y_train, y_test, complete_filename_ANN)
    # imp_ANN.ANN_gridsearch_mechanoadaptation_vs_passive_elliptic(X_train, X_test, y_train, y_test, complete_filename_ANN_size16)
    imp_ANN.ANN_gridsearch_mechanoadaptation_vs_passive_elliptic_new_settings(
        X_train, X_test, y_train, y_test, complete_filename_ANN_size4_article
    )


def build_predictions_Kriging(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample):
    uptake.metamodel_implementation.metamodel_creation.metamodel_creation_routine_kriging_mechanoadaptation_vs_passive_elliptic(
        datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample
    )


def build_predictions_Kriging_new_settings(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample):
    uptake.metamodel_implementation.metamodel_creation.metamodel_creation_routine_kriging_mechanoadaptation_vs_passive_elliptic_new_settings(
        datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample
    )


def build_predictions_PCE(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample):
    degree = 5
    uptake.metamodel_implementation.metamodel_creation.metamodel_creation_routine_pce_mechanoadaptation_vs_passive_elliptic(
        datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree
    )


def build_predictions_PCE_new_settings(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample):
    degree = 5
    uptake.metamodel_implementation.metamodel_creation.metamodel_creation_routine_pce_mechanoadaptation_vs_passive_elliptic_new_settings(
        datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree
    )


def build_all_predictions(
    datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, X_train, X_test, y_train, y_test
):
    build_predictions_ANN(X_train, X_test, y_train, y_test)
    build_predictions_Kriging(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample)
    build_predictions_PCE(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample)


def build_all_predictions_new_settings(
    datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, X_train, X_test, y_train, y_test
):
    build_predictions_ANN_new_settings(X_train, X_test, y_train, y_test)
    build_predictions_Kriging_new_settings(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample)
    build_predictions_PCE_new_settings(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample)

def extract_predictions_ANN(X_train, X_test, y_train, y_test):
    complete_filename_ANN = "predictions_ANN_mechanoadaptation_vs_passive_elliptic.pkl"
    _, y_pred_ANN, Q2_ANN = ANN_miu.extract_predictions(complete_filename_ANN)
    return y_pred_ANN, Q2_ANN


def extract_predictions_size16_ANN(X_train, X_test, y_train, y_test):
    complete_filename_ANN_size16 = "predictions_ANN_size16_mechanoadaptation_vs_passive_elliptic.pkl"
    _, y_pred_ANN, Q2_ANN = ANN_miu.extract_predictions(complete_filename_ANN_size16)
    return y_pred_ANN, Q2_ANN


def extract_predictions_size16_ANN_new_settings(X_train, X_test, y_train, y_test):
    complete_filename_ANN_size16 = "predictions_ANN_mechanoadaptation_vs_passive_elliptic_new_settings-v2.pkl"
    _, y_pred_ANN, Q2_ANN = ANN_miu.extract_predictions(complete_filename_ANN_size16)
    return y_pred_ANN, Q2_ANN

def extract_predictions_size4_ANN_new_settings_article(X_train, X_test, y_train, y_test):
    complete_filename_ANN_size4 = "predictions_ANN_mechanoadaptation_vs_passive_elliptic_new_settings-v2.pkl"
    _, y_pred_ANN, Q2_ANN = ANN_miu.extract_predictions(complete_filename_ANN_size4)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(y_test, y_pred_ANN, 'o')
    return y_pred_ANN, Q2_ANN

def extract_predictions_Kriging(
    X_test, y_test, datapresetting, metamodelvalidation, metamodelposttreatment, shuffled_sample
):
    complete_pkl_filename_Kriging = miu.create_pkl_name(
        "Kriging_mechanoadaptation_vs_passive_elliptic", datapresetting.training_amount
    )
    _, results_from_algo_Kriging = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_Kriging)
    metamodel_Kriging = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_Kriging)
    y_pred_Kriging = metamodel_Kriging(X_test)
    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(X_test, y_test, metamodel_Kriging)
    Q2_Kriging = metamodel_validator.computePredictivityFactor()
    return y_pred_Kriging, Q2_Kriging


def extract_predictions_Kriging_new_settings(
    X_test, y_test, datapresetting, metamodelvalidation, metamodelposttreatment, shuffled_sample
):
    complete_pkl_filename_Kriging = miu.create_pkl_name(
        "Kriging_mechanoadaptation_vs_passive_elliptic_new_settings_article", datapresetting.training_amount
    )
    _, results_from_algo_Kriging = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_Kriging)
    metamodel_Kriging = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_Kriging)
    y_pred_Kriging = metamodel_Kriging(X_test)
    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(X_test, y_test, metamodel_Kriging)
    Q2_Kriging = metamodel_validator.computePredictivityFactor()
    return y_pred_Kriging, Q2_Kriging


def extract_predictions_PCE(X_test, datapresetting, metamodelvalidation, metamodelposttreatment, shuffled_sample):
    degree = 5
    complete_pkl_filename_PCE = miu.create_pkl_name(
        "PCE_mechanoadaptation_vs_passive_elliptic" + str(degree), datapresetting.training_amount
    )
    _, results_from_algo_PCE = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_PCE)
    metamodel_PCE = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_PCE)
    gamma_bar_0_list_rescaled = miu.rescale_sample(X_test[:, 0])
    sigma_bar_list_rescaled = miu.rescale_sample(X_test[:, 1])
    gamma_bar_r_list_rescaled = miu.rescale_sample(X_test[:, 2])
    gamma_bar_fs_list_rescaled = miu.rescale_sample(X_test[:, 3])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(X_test[:, 4])
    r_bar_list_rescaled = miu.rescale_sample(X_test[:, 5])
    X_test_rescaled = ot.Sample(len(gamma_bar_0_list_rescaled), 6)
    for k in range(len(gamma_bar_0_list_rescaled)):
        X_test_rescaled[k, 0] = gamma_bar_0_list_rescaled[k]
        X_test_rescaled[k, 1] = sigma_bar_list_rescaled[k]
        X_test_rescaled[k, 2] = gamma_bar_r_list_rescaled[k]
        X_test_rescaled[k, 3] = gamma_bar_fs_list_rescaled[k]
        X_test_rescaled[k, 4] = gamma_bar_lambda_list_rescaled[k]
        X_test_rescaled[k, 5] = r_bar_list_rescaled[k]
    y_pred_PCE = metamodel_PCE(X_test_rescaled)
    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(X_test_rescaled, y_test, metamodel_PCE)
    Q2_PCE = metamodel_validator.computePredictivityFactor()
    return y_pred_PCE, Q2_PCE


def extract_predictions_PCE_new_settings(
    X_test, datapresetting, metamodelvalidation, metamodelposttreatment, shuffled_sample
):
    degree = 5
    complete_pkl_filename_PCE = miu.create_pkl_name(
        "PCE_mechanoadaptation_vs_passive_elliptic_new_settings_article" + str(degree), datapresetting.training_amount
    )
    _, results_from_algo_PCE = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_PCE)
    metamodel_PCE = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_PCE)
    gamma_bar_0_list_rescaled = miu.rescale_sample(X_test[:, 0])
    sigma_bar_list_rescaled = miu.rescale_sample(X_test[:, 1])
    gamma_bar_r_list_rescaled = miu.rescale_sample(X_test[:, 2])
    gamma_bar_fs_list_rescaled = miu.rescale_sample(X_test[:, 3])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(X_test[:, 4])
    r_bar_list_rescaled = miu.rescale_sample(X_test[:, 5])
    X_test_rescaled = ot.Sample(len(gamma_bar_0_list_rescaled), 6)
    for k in range(len(gamma_bar_0_list_rescaled)):
        X_test_rescaled[k, 0] = gamma_bar_0_list_rescaled[k]
        X_test_rescaled[k, 1] = sigma_bar_list_rescaled[k]
        X_test_rescaled[k, 2] = gamma_bar_r_list_rescaled[k]
        X_test_rescaled[k, 3] = gamma_bar_fs_list_rescaled[k]
        X_test_rescaled[k, 4] = gamma_bar_lambda_list_rescaled[k]
        X_test_rescaled[k, 5] = r_bar_list_rescaled[k]
    y_pred_PCE = metamodel_PCE(X_test_rescaled)
    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(X_test_rescaled, y_test, metamodel_PCE)
    Q2_PCE = metamodel_validator.computePredictivityFactor()
    return y_pred_PCE, Q2_PCE


def extract_all_predictions(
    datapresetting, metamodelvalidation, metamodelposttreatment, shuffled_sample, X_train, X_test, y_train, y_test
):
    y_pred_ANN, Q2_ANN = extract_predictions_size16_ANN(X_train, X_test, y_train, y_test)
    y_pred_Kriging, Q2_Kriging = extract_predictions_Kriging(
        X_test, y_test, datapresetting, metamodelvalidation, metamodelposttreatment, shuffled_sample
    )
    y_pred_PCE, Q2_PCE = extract_predictions_PCE(
        X_test, datapresetting, metamodelvalidation, metamodelposttreatment, shuffled_sample
    )
    return y_pred_ANN, y_pred_Kriging, y_pred_PCE, Q2_ANN, Q2_Kriging, Q2_PCE


def extract_all_predictions_new_settings(
    datapresetting, metamodelvalidation, metamodelposttreatment, shuffled_sample, X_train, X_test, y_train, y_test
):
    y_pred_ANN, Q2_ANN = extract_predictions_size4_ANN_new_settings_article(X_train, X_test, y_train, y_test)
    y_pred_Kriging, Q2_Kriging = extract_predictions_Kriging_new_settings(
        X_test, y_test, datapresetting, metamodelvalidation, metamodelposttreatment, shuffled_sample
    )
    y_pred_PCE, Q2_PCE = extract_predictions_PCE_new_settings(
        X_test, datapresetting, metamodelvalidation, metamodelposttreatment, shuffled_sample
    )
    return y_pred_ANN, y_pred_Kriging, y_pred_PCE, Q2_ANN, Q2_Kriging, Q2_PCE


def plot_comparison_predictions_surrogates(
    y_test, y_pred_ANN, y_pred_Kriging, y_pred_PCE, Q2_ANN, Q2_Kriging, Q2_PCE, createfigure, savefigure, fonts, pixels
):
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    green = palette[3]
    color_plot_PCE = orange
    color_plot_Kriging = purple
    color_plot_ANN = green
    ax.plot(
        y_test,
        y_pred_PCE,
        "o",
        color=color_plot_PCE,
        alpha=0.6,
        label=r"PCE: $Q_2^{PCE}$ = " + str(np.round(Q2_PCE[0], 2)),
    )
    ax.plot(
        y_test,
        y_pred_Kriging,
        "o",
        color=color_plot_Kriging,
        alpha=0.6,
        label=r"Kriging: $Q_2^{KRI}$ = " + str(np.round(Q2_Kriging[0], 2)),
    )
    ax.plot(
        y_test,
        y_pred_ANN,
        "o",
        color=color_plot_ANN,
        alpha=0.7,
        label=r"ANN: $Q_2^{ANN}$ = " + str(np.round(Q2_ANN, 2)),
    )
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
    ax.legend(prop=fonts.serif(), loc="upper left", framealpha=0.7)
    ax.set_xlabel(r"true values of $\tilde{F}$", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"predicted values of $\tilde{F}$", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "comparison_surrogates_mechanoadaptation_vs_passive_elliptic_size16_" + str(pixels))


def plot_comparison_predictions_surrogates_new_settings(
    y_test, y_pred_ANN, y_pred_Kriging, y_pred_PCE, Q2_ANN, Q2_Kriging, Q2_PCE, createfigure, savefigure, fonts, pixels
):
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    green = palette[3]
    color_plot_PCE = orange
    color_plot_Kriging = purple
    color_plot_ANN = green
    # ax.plot(
    #     y_test,
    #     y_pred_PCE,
    #     "o",
    #     color=color_plot_PCE,
    #     alpha=0.6,
    #     label=r"$Q_2^{PCE}$ = " + str(np.round(Q2_PCE[0], 2)),
    # )
    ax.plot(
        y_test,
        y_pred_Kriging,
        "o",
        color=color_plot_Kriging,
        alpha=0.6,
        label=r"$Q_2^{KRI}$ = " + str(np.round(Q2_Kriging[0], 2)),
    )
    # ax.plot(
    #     y_test,
    #     y_pred_ANN,
    #     "o",
    #     color=color_plot_ANN,
    #     alpha=0.7,
    #     label=r"ANN: $Q_2^{ANN}$ = " + str(np.round(Q2_ANN, 2)),
    # )
    ax.plot([0, 1], [0, 1], "-k", linewidth=2)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(
        [0, 0.2, 0.4, 0.6, 0.8, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(
        [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-0.02, 1.02))
    ax.set_ylim((-0.22, 1.02))
    ax.grid(linestyle="--")
    ax.legend(prop=fonts.serif(), loc="upper left", framealpha=0.7)
    ax.set_xlabel(r"true values of $\tilde{F}$", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"predicted values of $\tilde{F}$", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(
        fig, "KRI_true_predicted_mechanoadaptation_vs_passive_elliptic_new_settings_size4_article_" + str(pixels)
    )


def plot_comparison_PDFs_surrogates(input_sample_training, y_train, createfigure, savefigure, fonts, pixels):

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
    experiment_input = ot.MonteCarloExperiment(distribution_input, int(1e3))
    sample_input_MC = experiment_input.generate()
    degree = 5
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
    complete_filename_grid = "grid_search.pkl"
    sc_X, grid = ANN_miu.extract_gridsearch(complete_filename_grid)
    # sc_X = StandardScaler()
    X_testscaled = sc_X.fit_transform(sample_input_MC)
    output_ANN_before_reshape = grid.predict(X_testscaled)
    (size_output_ANN_before_reshape,) = np.shape(output_ANN_before_reshape)
    output_ANN = ot.Sample(size_output_ANN_before_reshape, 1)
    for i in range(size_output_ANN_before_reshape):
        output_ANN[i, 0] = output_ANN_before_reshape[i]
    X_plot = np.linspace(0, 1, 2000)[:, None]

    kde_model = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_model)
    log_dens_model = kde_model.score_samples(X_plot)
    kde_kriging = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_kriging)
    log_dens_kriging = kde_kriging.score_samples(X_plot)
    kde_pce = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_pce)
    log_dens_pce = kde_pce.score_samples(X_plot)
    kde_ANN = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_ANN)
    log_dens_ANN = kde_ANN.score_samples(X_plot)

    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    ax.hist(output_model, bins=20, density=True, color="gray", alpha=0.3, ec="black")

    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    green = palette[3]

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
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_ANN),
        color=green,
        lw=4,
        linestyle="-.",
        label="ANN",
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
    savefigure.save_as_png(fig, "PDF_metamodel_ANN_mechanoadaptation_vs_passive_elliptic_size16" + str(pixels))


def plot_comparison_PDFs_surrogates_new_settings(
    input_sample_training, y_train, createfigure, savefigure, fonts, pixels
):

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
    experiment_input = ot.MonteCarloExperiment(distribution_input, int(1e3))
    sample_input_MC = experiment_input.generate()
    degree = 5
    complete_pkl_filename_pce = miu.create_pkl_name(
        "PCE_mechanoadaptation_vs_passive_elliptic_new_settings_article" + str(degree), training_amount
    )
    shuffled_sample, results_from_algo_pce = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_pce)
    metamodel_pce = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_pce)
    complete_pkl_filename_kriging = miu.create_pkl_name(
        "Kriging_mechanoadaptation_vs_passive_elliptic_new_settings_article", training_amount
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
    output_model_hist = [x[0] for x in output_model]
    output_pce = metamodel_pce(input_sample_rescaled)
    output_kriging = metamodel_kriging(input_sample)
    complete_filename_grid = "grid_search_article-v2.pkl"
    sc_X, grid = ANN_miu.extract_gridsearch(complete_filename_grid)
    # sc_X = StandardScaler()
    X_testscaled = sc_X.fit_transform(sample_input_MC)
    output_ANN_before_reshape = grid.predict(X_testscaled)
    (size_output_ANN_before_reshape,) = np.shape(output_ANN_before_reshape)
    output_ANN = ot.Sample(size_output_ANN_before_reshape, 1)
    for i in range(size_output_ANN_before_reshape):
        output_ANN[i, 0] = output_ANN_before_reshape[i]
    X_plot = np.linspace(0, 1, 2000)[:, None]

    kde_model = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_model)
    log_dens_model = kde_model.score_samples(X_plot)
    kde_kriging = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_kriging)
    log_dens_kriging = kde_kriging.score_samples(X_plot)
    kde_pce = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_pce)
    log_dens_pce = kde_pce.score_samples(X_plot)
    kde_ANN = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(output_ANN)
    log_dens_ANN = kde_ANN.score_samples(X_plot)
    palette_set2 = sns.color_palette("Set2")
    lightgray = palette_set2[-1]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    ax.hist(output_model_hist, bins=20, color=lightgray,density=True, alpha=0.3, ec="black")

    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    green = palette[3]

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
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens_ANN),
        color=green,
        lw=4,
        linestyle="-.",
        label="ANN",
    )
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(
        [0, 0.2, 0.4, 0.6, 0.8, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 2, 4, 6,8, 10])
    ax.set_yticklabels(
        [0, 2, 4, 6,8, 10],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-0.02, 1.02))
    ax.set_ylim((0, 11))
    ax.set_xlabel(r"$\tilde{f}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$p_{\tilde{F}}(\tilde{f})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax.grid(linestyle="--")
    savefigure.save_as_png(
        fig, "PDF_metamodel_ANN_mechanoadaptation_vs_passive_elliptic_new_settings_size4_article-v2" + str(pixels)
    )



def define_test_samples_for_comparison():
    filename = "dataset_for_ANN_mechanadaptation_vs_passive_elliptic.txt"
    training_amount = 0.9
    datapresetting = DataPreSetting(filename, training_amount)
    shuffled_sample = datapresetting.shuffle_dataset_from_datafile()
    X_train, y_train = datapresetting.extract_training_data_from_shuffled_dataset_mechanoadaptation_vs_passive_elliptic(
        shuffled_sample
    )
    X_test, y_test = datapresetting.extract_testing_data_mechanoadaptation_vs_passive_elliptic(shuffled_sample)
    ANN_miu.export_settings_for_metamodels(
        datapresetting, training_amount, shuffled_sample, X_train, X_test, y_train, y_test
    )


def define_test_samples_for_comparison_new_settings():
    filename = "dataset_for_ANN_mechanadaptation_vs_passive_elliptic_newsettings_size4.txt"
    training_amount = 0.9
    datapresetting = DataPreSetting(filename, training_amount)
    shuffled_sample = datapresetting.shuffle_dataset_from_datafile()
    X_train, y_train = datapresetting.extract_training_data_from_shuffled_dataset_mechanoadaptation_vs_passive_elliptic(
        shuffled_sample
    )
    X_test, y_test = datapresetting.extract_testing_data_mechanoadaptation_vs_passive_elliptic(shuffled_sample)
    ANN_miu.export_settings_for_metamodels_new_settings(
        datapresetting, training_amount, shuffled_sample, X_train, X_test, y_train, y_test
    )


if __name__ == "__main__":
    fonts = Fonts()
    createfigure = CreateFigure()
    savefigure = SaveFigure()
    pixels = 360
    metamodelposttreatment = MetamodelPostTreatment()

    # define_test_samples_for_comparison_new_settings()
    size = 4
    (
        datapresetting,
        training_amount,
        shuffled_sample,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = ANN_miu.extract_settings_for_comparison_surrogates_new_settings(size)
    metamodelcreation = MetamodelCreation(X_train, y_train)
    metamodelvalidation = MetamodelValidation()
    
    
    # build_all_predictions_new_settings(
    #     datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, X_train, X_test, y_train, y_test
    # )
    y_pred_ANN, y_pred_Kriging, y_pred_PCE, Q2_ANN, Q2_Kriging, Q2_PCE = extract_all_predictions_new_settings(
        datapresetting, metamodelvalidation, metamodelposttreatment, shuffled_sample, X_train, X_test, y_train, y_test
    )

    plot_comparison_predictions_surrogates_new_settings(
        y_test,
        y_pred_ANN,
        y_pred_Kriging,
        y_pred_PCE,
        Q2_ANN,
        Q2_Kriging,
        Q2_PCE,
        createfigure,
        savefigure,
        fonts,
        pixels,
    )

    # plot_comparison_PDFs_surrogates_new_settings(X_train, y_train, createfigure, savefigure, fonts, pixels)
