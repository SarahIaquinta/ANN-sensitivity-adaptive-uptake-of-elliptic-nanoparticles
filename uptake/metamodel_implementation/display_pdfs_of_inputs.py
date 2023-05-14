import pickle
from pathlib import Path

import numpy as np
import openturns as ot
import seaborn as sns

ot.Log.Show(ot.Log.NONE)

import matplotlib.pyplot as plt
import scipy.stats
from scipy.interpolate import UnivariateSpline
from sklearn.neighbors import KernelDensity

from uptake.figures.utils import CreateFigure, Fonts, SaveFigure, XTickLabels, XTicks
from uptake.metamodel_implementation.metamodel_creation import DataPreSetting, MetamodelPostTreatment


# def plot_PDF_sample_inputs(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
#     palette = sns.color_palette("Set2")
#     vert, bleu, jaune = palette[0], palette[2], palette[5]
#     gamma_bar_0_list = input_data[:, 0]
#     sigma_bar_list = input_data[:, 1]
#     r_bar_list = input_data[:, 2]
#     fig = createfigure.square_figure_7(pixels=pixels)
#     ax = fig.gca()
#     ax.hist(gamma_bar_0_list, bins=nb_bin, color=vert, ec="black")
#     ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
#     ax.set_xticklabels(
#         [1, 2, 3, 4, 5, 6, 7, 8],
#         font=fonts.serif(),
#         fontsize=fonts.axis_legend_size(),
#     )
#     ax.set_yticks([0, 10, 20, 30])
#     ax.set_yticklabels(
#         [0, 10, 20, 30],
#         font=fonts.serif(),
#         fontsize=fonts.axis_legend_size(),
#     )
#     ax.set_xlim((0.9, 8.1))
#     ax.set_ylim((0, 31))
#     ax.set_xlabel(r"$\overline{\Gamma}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
#     ax.set_ylabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
#     savefigure.save_as_png(fig, "PDF_gamma_bar_feq" + str(pixels))

#     fig = createfigure.square_figure_7(pixels=pixels)
#     ax = fig.gca()
#     ax.hist(sigma_bar_list, bins=nb_bin, color=bleu, ec="black")
#     ax.set_xticks([1, 2, 3, 4, 5, 6])
#     ax.set_xticklabels(
#         [1, 2, 3, 4, 5, 6],
#         font=fonts.serif(),
#         fontsize=fonts.axis_legend_size(),
#     )
#     ax.set_yticks([0, 10, 20, 30])
#     ax.set_yticklabels(
#         [0, 10, 20, 30],
#         font=fonts.serif(),
#         fontsize=fonts.axis_legend_size(),
#     )
#     ax.set_xlim((0.4, 5.6))
#     ax.set_ylim((0, 31))
#     ax.set_xlabel(r"$\overline{\Sigma}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
#     ax.set_ylabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
#     savefigure.save_as_png(fig, "PDF_sigma_bar_feq" + str(pixels))

#     fig = createfigure.square_figure_7(pixels=pixels)
#     ax = fig.gca()
#     ax.hist(r_bar_list, bins=nb_bin, color=jaune, ec="black")
#     ax.set_xticks([1 / 6, 1, 6])
#     ax.set_xticklabels(
#         ["1/6", "1", "6"],
#         font=fonts.serif(),
#         fontsize=fonts.axis_legend_size(),
#     )
#     ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
#     ax.set_yticklabels(
#         [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
#         font=fonts.serif(),
#         fontsize=fonts.axis_legend_size(),
#     )
#     ax.set_xlim((0, 6.17))
#     ax.set_ylim((0, 92))
#     ax.set_xlabel(r"$\overline{R}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
#     ax.set_ylabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
#     savefigure.save_as_png(fig, "PDF_r_bar_feq" + str(pixels))


def rescale_sample(vector):
    vector_start0 = [k[0] - vector.getMin()[0] for k in vector]
    vector_end2 = [k * 2 / max(vector_start0) for k in vector_start0]
    vector_normalized = [k - 1 for k in vector_end2]
    return vector_normalized


def plot_PDF_sample_inputs_normalized(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    vert_clair, gris, beige = palette[2], palette[0], palette[-2]
    gamma_bar_0_list = input_data[:, 0]
    sigma_bar_list = input_data[:, 1]
    gamma_bar_A_list = input_data[:, 2]
    gamma_bar_D_list = input_data[:, 3]
    gamma_bar_S_list = input_data[:, 4]
    r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()

    gamma_bar_0_list_normalized = rescale_sample(gamma_bar_0_list)
    ax.hist(gamma_bar_0_list_normalized, bins=nb_bin, color=vert_clair, ec="black")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(
        [-1, 0, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 10, 20, 30])
    ax.set_yticklabels(
        [0, 10, 20, 30],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((0, 31))
    ax.set_xlabel(r"$\overline{\Gamma}^*$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_gamma_bar_normalized_feq" + str(pixels))

    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    sigma_bar_list_normalized = rescale_sample(sigma_bar_list)
    ax.hist(sigma_bar_list_normalized, bins=nb_bin, color=gris, ec="black")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(
        [-1, 0, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 10, 20, 30])
    ax.set_yticklabels(
        [0, 10, 20, 30],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((0, 31))
    ax.set_xlabel(r"$\overline{\Sigma}^*$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_sigma_bar_normalized_feq" + str(pixels))

    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    r_bar_list_normalized = rescale_sample(r_bar_list)
    ax.hist(r_bar_list_normalized, bins=nb_bin, color=beige, ec="black")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(
        [-1, 0, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 10, 20, 30, 40, 60, 70, 80, 90])
    ax.set_yticklabels(
        [0, 10, 20, 30, 40, 60, 70, 80, 90],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((0, 91))
    ax.set_xlabel(r"$\overline{R}^*$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_r_bar_normalized_feq" + str(pixels))

def plot_PDF_gamma_bar_0(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    colors = [palette[k] for k in range(1, len(palette), 2)]
    color_gamma_bar_0 = colors[0]
    # vert_clair, gris, beige = palette[2], palette[0], palette[-2]
    gamma_bar_0_list = input_data[:, 0]
    # sigma_bar_list = input_data[:, 1]
    # gamma_bar_A_list = input_data[:, 2]
    # gamma_bar_D_list = input_data[:, 3]
    # gamma_bar_S_list = input_data[:, 4]
    # r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    X_plot_gamma_bar_0 = np.linspace(0.8, 8.2, 2000)[:, None]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(gamma_bar_0_list)
    log_dens_model = kde_model.score_samples(X_plot_gamma_bar_0)
    ax.hist(gamma_bar_0_list, bins=nb_bin, density=True, color=color_gamma_bar_0, alpha=0.6, ec="black")
    # ax.plot(
    #     X_plot_gamma_bar_0[:, 0],
    #     np.exp(log_dens_model),
    #     color='k',
    #     lw=2,
    #     linestyle="-", label="model",
    # )
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_xticklabels(
        [1, 2, 3, 4, 5, 6, 7, 8],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14])
    ax.set_yticklabels(
        [0, 2, 4, 6, 8, 10, 12, 14],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((0.8, 8.2))
    ax.set_ylim((0, 0.15))
    ax.set_xlabel(r"$\overline{\gamma}_0$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"Distribution of $\overline{\Gamma}_{0} ~ (\times 10^{-2})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_gamma_bar_0-v2")

def plot_PDF_gamma_bar_0_new_settings(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    colors = [palette[k] for k in range(1, len(palette), 2)]
    color_gamma_bar_0 = colors[0]
    # vert_clair, gris, beige = palette[2], palette[0], palette[-2]
    gamma_bar_0_list = input_data[:, 0]
    # sigma_bar_list = input_data[:, 1]
    # gamma_bar_A_list = input_data[:, 2]
    # gamma_bar_D_list = input_data[:, 3]
    # gamma_bar_S_list = input_data[:, 4]
    # r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    X_plot_gamma_bar_0 = np.linspace(0.8, 8.2, 2000)[:, None]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(gamma_bar_0_list)
    log_dens_model = kde_model.score_samples(X_plot_gamma_bar_0)
    ax.hist(gamma_bar_0_list, bins=nb_bin, density=True, color=color_gamma_bar_0, alpha=0.6, ec="black")
    # ax.plot(
    #     X_plot_gamma_bar_0[:, 0],
    #     np.exp(log_dens_model),
    #     color='k',
    #     lw=2,
    #     linestyle="-", label="model",
    # )
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_xticklabels(
        [1, 2, 3, 4, 5, 6, 7, 8],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14])
    ax.set_yticklabels(
        [0, 2, 4, 6, 8, 10, 12, 14],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((0.8, 8.2))
    ax.set_ylim((0, 0.15))
    ax.set_xlabel(r"$\overline{\gamma}_0$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"Distribution of $\overline{\Gamma}_{0} ~ (\times 10^{-2})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_gamma_bar_0_new_settings")

def plot_PDF_sigma_bar(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    colors = [palette[k] for k in range(1, len(palette), 2)]
    color_sigma_bar = colors[1]
    # vert_clair, gris, beige = palette[2], palette[0], palette[-2]
    # gamma_bar_0_list = input_data[:, 0]
    sigma_bar_list = input_data[:, 1]
    # gamma_bar_A_list = input_data[:, 2]
    # gamma_bar_D_list = input_data[:, 3]
    # gamma_bar_S_list = input_data[:, 4]
    # r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    X_plot_sigma_bar = np.linspace(0.3, 5.7, 2000)[:, None]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(sigma_bar_list)
    log_dens_model = kde_model.score_samples(X_plot_sigma_bar)
    ax.hist(sigma_bar_list, bins=nb_bin, density=True, color=color_sigma_bar, alpha=0.6, ec="black")
    # ax.plot(
    #     X_plot_sigma_bar[:, 0],
    #     np.exp(log_dens_model),
    #     color='k',
    #     lw=2,
    #     linestyle="-", label="model",
    # )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(
        [1, 2, 3, 4, 5],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
    ax.set_yticklabels(
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((0.3, 5.7))
    ax.set_ylim((0, 0.21))
    ax.set_xlabel(r"$\overline{\sigma}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"Distribution of $\overline{\Sigma} ~ (\times 10^{-2})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_sigma_bar-v2")

def plot_PDF_sigma_bar_new_settings(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    colors = [palette[k] for k in range(1, len(palette), 2)]
    color_sigma_bar = colors[1]
    # vert_clair, gris, beige = palette[2], palette[0], palette[-2]
    # gamma_bar_0_list = input_data[:, 0]
    sigma_bar_list = input_data[:, 1]
    # gamma_bar_A_list = input_data[:, 2]
    # gamma_bar_D_list = input_data[:, 3]
    # gamma_bar_S_list = input_data[:, 4]
    # r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    X_plot_sigma_bar = np.linspace(0.3, 5.7, 2000)[:, None]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(sigma_bar_list)
    log_dens_model = kde_model.score_samples(X_plot_sigma_bar)
    ax.hist(sigma_bar_list, bins=nb_bin, density=True, color=color_sigma_bar, alpha=0.6, ec="black")
    # ax.plot(
    #     X_plot_sigma_bar[:, 0],
    #     np.exp(log_dens_model),
    #     color='k',
    #     lw=2,
    #     linestyle="-", label="model",
    # )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(
        [1, 2, 3, 4, 5],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
    ax.set_yticklabels(
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((0.3, 5.7))
    ax.set_ylim((0, 0.21))
    ax.set_xlabel(r"$\overline{\sigma}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"Distribution of $\overline{\Sigma}_A ~ (\times 10^{-2})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_sigma_bar_new_settings")

def plot_PDF_gamma_bar_A_new_settings(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    colors = [palette[k] for k in range(1, len(palette), 2)]
    color_gamma_bar_A = colors[2]
    # gamma_bar_0_list = input_data[:, 0]
    # sigma_bar_list = input_data[:, 1]
    gamma_bar_A_list = input_data[:, 2]
    # gamma_bar_D_list = input_data[:, 3]
    # gamma_bar_S_list = input_data[:, 4]
    # r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    X_plot_gamma_bar_A = np.linspace(0.8, 6.2, 2000)[:, None]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(gamma_bar_A_list)
    log_dens_model = kde_model.score_samples(X_plot_gamma_bar_A)
    ax.hist(gamma_bar_A_list, bins=nb_bin, density=True, color=color_gamma_bar_A, alpha=0.6, ec="black")
    # ax.plot(
    #     X_plot_gamma_bar_A[:, 0],
    #     np.exp(log_dens_model),
    #     color='k',
    #     lw=2,
    #     linestyle="-", label="model",
    # )
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(
        [1, 2, 3, 4, 5, 6],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
    ax.set_yticklabels(
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((0.8, 6.2))
    ax.set_ylim((0, 0.21))
    ax.set_xlabel(r"$\overline{\gamma}_A$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"Distribution of $\overline{\Gamma}_{A} ~ (\times 10^{-2})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_gamma_bar_A_new_settings")

def plot_PDF_gamma_bar_A(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    colors = [palette[k] for k in range(1, len(palette), 2)]
    color_gamma_bar_A = colors[2]
    # gamma_bar_0_list = input_data[:, 0]
    # sigma_bar_list = input_data[:, 1]
    gamma_bar_A_list = input_data[:, 2]
    # gamma_bar_D_list = input_data[:, 3]
    # gamma_bar_S_list = input_data[:, 4]
    # r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    X_plot_gamma_bar_A = np.linspace(0.8, 6.2, 2000)[:, None]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(gamma_bar_A_list)
    log_dens_model = kde_model.score_samples(X_plot_gamma_bar_A)
    ax.hist(gamma_bar_A_list, bins=nb_bin, density=True, color=color_gamma_bar_A, alpha=0.6, ec="black")
    # ax.plot(
    #     X_plot_gamma_bar_A[:, 0],
    #     np.exp(log_dens_model),
    #     color='k',
    #     lw=2,
    #     linestyle="-", label="model",
    # )
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(
        [1, 2, 3, 4, 5, 6],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
    ax.set_yticklabels(
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((0.8, 6.2))
    ax.set_ylim((0, 0.21))
    ax.set_xlabel(r"$\overline{\gamma}_A$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"Distribution of $\overline{\Gamma}_{A} ~ (\times 10^{-2})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_gamma_bar_A-v2")

def plot_PDF_gamma_bar_D_new_settings(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    colors = [palette[k] for k in range(1, len(palette), 2)]
    color_gamma_bar_D = colors[3]    # vert_clair, gris, beige = palette[2], palette[0], palette[-2]
    # gamma_bar_0_list = input_data[:, 0]
    # sigma_bar_list = input_data[:, 1]
    # gamma_bar_A_list = input_data[:, 2]
    gamma_bar_D_list = input_data[:, 3]
    # gamma_bar_S_list = input_data[:, 4]
    # r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    X_plot_gamma_bar_A = np.linspace(-0.47, 0.47, 2000)[:, None]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(gamma_bar_D_list)
    log_dens_model = kde_model.score_samples(X_plot_gamma_bar_A)
    ax.hist(gamma_bar_D_list, bins=nb_bin, density=True, color=color_gamma_bar_D, alpha=0.6, ec="black")
    # ax.plot(
    #     X_plot_gamma_bar_A[:, 0],
    #     np.exp(log_dens_model),
    #     color='k',
    #     lw=2,
    #     linestyle="-", label="model",
    # )
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xticklabels(
        [-0.2, -0.1, 0, 0.1, 0.2],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(
        [0, 20, 40, 60, 80, 100],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-0.47, 0.47))
    ax.set_ylim((0, 1.16))
    ax.set_xlabel(r"$\overline{\gamma}_D$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"Distribution of $\overline{\Gamma}_D ~ (\times 10^{-2})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_gamma_bar_D_new_settings")

def plot_PDF_gamma_bar_D(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    colors = [palette[k] for k in range(1, len(palette), 2)]
    color_gamma_bar_D = colors[3]    
    # sigma_bar_list = input_data[:, 1]
    # gamma_bar_A_list = input_data[:, 2]
    gamma_bar_D_list = input_data[:, 3]
    # gamma_bar_S_list = input_data[:, 4]
    # r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    X_plot_gamma_bar_A = np.linspace(-0.47, 0.47, 2000)[:, None]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(gamma_bar_D_list)
    log_dens_model = kde_model.score_samples(X_plot_gamma_bar_A)
    ax.hist(gamma_bar_D_list, bins=nb_bin, density=True, color=color_gamma_bar_D, alpha=0.6, ec="black")
    # ax.plot(
    #     X_plot_gamma_bar_A[:, 0],
    #     np.exp(log_dens_model),
    #     color='k',
    #     lw=2,
    #     linestyle="-", label="model",
    # )
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xticklabels(
        [-0.2, -0.1, 0, 0.1, 0.2],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(
        [0, 20, 40, 60, 80, 100],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-0.47, 0.47))
    ax.set_ylim((0, 1.16))
    ax.set_xlabel(r"$\overline{\gamma}_D$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"Distribution of $\overline{\Gamma}_D ~ (\times 10^{-2})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_gamma_bar_D-v2")

def plot_PDF_gamma_bar_S_new_settings(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    colors = [palette[k] for k in range(1, len(palette), 2)]
    color_gamma_bar_S = colors[4]    # vert_clair, gris, beige = palette[2], palette[0], palette[-2]
    # gamma_bar_0_list = input_data[:, 0]
    # sigma_bar_list = input_data[:, 1]
    # gamma_bar_A_list = input_data[:, 2]
    # gamma_bar_D_list = input_data[:, 3]
    gamma_bar_S_list = input_data[:, 4]
    # r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    X_plot_gamma_bar_D = np.linspace(8, 52, 2000)[:, None]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(gamma_bar_S_list)
    log_dens_model = kde_model.score_samples(X_plot_gamma_bar_D)
    ax.hist(gamma_bar_S_list, bins=nb_bin, density=True, color=color_gamma_bar_S, alpha=0.6, ec="black")
    # ax.plot(
    #     X_plot_gamma_bar_D[:, 0],
    #     np.exp(log_dens_model),
    #     color='k',
    #     lw=2,
    #     linestyle="-", label="model",
    # )
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_xticklabels(
        [10, 20, 30, 40, 50],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.005, 0.01, 0.015, 0.02, 0.025])
    ax.set_yticklabels(
        [0, 5, 10, 15, 20, 25],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((8, 52))
    ax.set_ylim((0, 0.027))
    ax.set_xlabel(r"$\overline{\gamma}_S$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"Distribution of $\overline{\Gamma}_S ~ (\times 10^{-3})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_gamma_bar_S_new_settings")

def plot_PDF_gamma_bar_S(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    colors = [palette[k] for k in range(1, len(palette), 2)]
    color_gamma_bar_S = colors[4]        # gamma_bar_0_list = input_data[:, 0]
    # sigma_bar_list = input_data[:, 1]
    # gamma_bar_A_list = input_data[:, 2]
    # gamma_bar_D_list = input_data[:, 3]
    gamma_bar_S_list = input_data[:, 4]
    # r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    X_plot_gamma_bar_D = np.linspace(8, 102, 2000)[:, None]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(gamma_bar_S_list)
    log_dens_model = kde_model.score_samples(X_plot_gamma_bar_D)
    ax.hist(gamma_bar_S_list, bins=nb_bin, density=True, color=color_gamma_bar_S, alpha=0.6, ec="black")
    # ax.plot(
    #     X_plot_gamma_bar_D[:, 0],
    #     np.exp(log_dens_model),
    #     color='k',
    #     lw=2,
    #     linestyle="-", label="model",
    # )
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xticklabels(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.005, 0.01])
    ax.set_yticklabels(
        [0, 5, 10],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((8, 102))
    ax.set_ylim((0, 0.0125))
    ax.set_xlabel(r"$\overline{\gamma}_S$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"Distribution of $\overline{\Gamma}_S ~ (\times 10^{-3})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_gamma_bar_S-v2")


def plot_PDF_r_bar_new_settings(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    colors = [palette[k] for k in range(1, len(palette), 2)]
    color_r_bar = colors[5]    # vert_clair, gris, beige = palette[2], palette[0], palette[-2]
    # gamma_bar_0_list = input_data[:, 0]
    # sigma_bar_list = input_data[:, 1]
    # gamma_bar_A_list = input_data[:, 2]
    # gamma_bar_D_list = input_data[:, 3]
    # gamma_bar_S_list = input_data[:, 4]
    r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    X_plot_r_bar = np.linspace(-0.04, 6.2, 2000)[:, None]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.04).fit(r_bar_list)
    log_dens_model = kde_model.score_samples(X_plot_r_bar)
    ax.hist(r_bar_list, bins=nb_bin, density=True, color=color_r_bar, alpha=0.6, ec="black")
    ax.plot(
        X_plot_r_bar[:, 0],
        np.exp(log_dens_model),
        color='k',
        lw=2,
        linestyle="-", label="model",
    )
    ax.set_xticks([0.16, 1, 6])
    ax.set_xticklabels(
        ['1/6', '1', '6'],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(
        [0, 5, 10, 15, 20],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-0.04, 6.2))
    ax.set_ylim((0, 2.2))
    ax.set_xlabel(r"$\overline{r}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$p_{\overline{R}}(\overline{r}) (\times 10^{-1})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_r_bar_new_settings")

def plot_PDF_r_bar(input_data, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
    palette = sns.color_palette("Paired")
    colors = [palette[k] for k in range(1, len(palette), 2)]
    color_rbar = colors[5]   
    # gamma_bar_0_list = input_data[:, 0]
    # sigma_bar_list = input_data[:, 1]
    # gamma_bar_A_list = input_data[:, 2]
    # gamma_bar_D_list = input_data[:, 3]
    # gamma_bar_S_list = input_data[:, 4]
    r_bar_list = input_data[:, 5]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    X_plot_r_bar = np.linspace(-0.04, 6.2, 2000)[:, None]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.04).fit(r_bar_list)
    log_dens_model = kde_model.score_samples(X_plot_r_bar)
    ax.hist(r_bar_list, bins=nb_bin, density=True, color=color_rbar, alpha=0.6, ec="black")
    # ax.plot(
    #     X_plot_r_bar[:, 0],
    #     np.exp(log_dens_model),
    #     color='k',
    #     lw=2,
    #     linestyle="-", label="model",
    # )
    ax.set_xticks([0.16, 1, 6])
    ax.set_xticklabels(
        ['1/6', '1', '6'],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_yticklabels(
        [0, 1, 2, 3, 4, 5, 6],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((-0.04, 6.2))
    ax.set_ylim((0, 0.65))
    ax.set_xlabel(r"$\overline{r}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"Distribution of $\overline{R} ~ (\times 10^{-1})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "PDF_r_bar-v2")



if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    xticks = XTicks()
    xticklabels = XTickLabels()
    # filename_qMC_Sobol_feq = "dataset_for_metamodel_creation_feq.txt"
    filename_qMC_mechanoadaptation_vs_passive_elliptic_new_settings = "dataset_for_ANN_mechanadaptation_vs_passive_elliptic_newsettings.txt"
    # filename_qMC_mechanoadaptation_vs_passive_elliptic = "dataset_for_ANN_mechanadaptation_vs_passive_elliptic.txt"
    training_amount_list = [0.95]  # [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    metamodelposttreatment = MetamodelPostTreatment()
    for training_amount in training_amount_list:
        datapresetting_feq = DataPreSetting(filename_qMC_mechanoadaptation_vs_passive_elliptic_new_settings, training_amount)
        shuffled_sample_feq = datapresetting_feq.shuffle_dataset_from_datafile()
        (
            input_sample_training_feq,
            output_sample_training_feq,
        ) = datapresetting_feq.extract_training_data_from_shuffled_dataset_mechanoadaptation_vs_passive_elliptic(shuffled_sample_feq)

    # plot_PDF_gamma_bar_0(input_sample_training_feq, 20, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360)
    # plot_PDF_sigma_bar(input_sample_training_feq, 20, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360)
    # plot_PDF_gamma_bar_A(input_sample_training_feq, 20, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360)
    # plot_PDF_gamma_bar_D(input_sample_training_feq, 20, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360)
    # plot_PDF_gamma_bar_S(input_sample_training_feq, 20, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360)
    # plot_PDF_r_bar(input_sample_training_feq, 50, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360)

    # plot_PDF_gamma_bar_0_new_settings(input_sample_training_feq, 20, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360)
    # plot_PDF_sigma_bar_new_settings(input_sample_training_feq, 20, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360)
    plot_PDF_gamma_bar_A_new_settings(input_sample_training_feq, 20, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360)
    plot_PDF_gamma_bar_D_new_settings(input_sample_training_feq, 20, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360)
    plot_PDF_gamma_bar_S_new_settings(input_sample_training_feq, 20, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360)
    plot_PDF_r_bar_new_settings(input_sample_training_feq, 50, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360)
