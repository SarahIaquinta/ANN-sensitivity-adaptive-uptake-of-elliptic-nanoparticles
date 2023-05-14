import pickle
from pathlib import Path

import numpy as np
import openturns as ot
import seaborn as sns
from sklearn.neighbors import KernelDensity

ot.Log.Show(ot.Log.NONE)

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from uptake.figures.utils import CreateFigure, Fonts, SaveFigure, XTickLabels, XTicks


class SampleRepresentativeness:
    def __init__(self, filename, resampling_size, pixels):
        """
        Constructs all the necessary attributes for the DataPreSetting object.

        Parameters:
            ----------
            filename: string
                name of the .txt file from which the data will be extracted
            training_amount: float
                proportion (between 0 and 1) of the initial data used for training (the remaining
                data are used for testing)
            resampling_size: float
                number of shuffled samples of the dataset, used to evaluate the standard
                deviation of the cumulative mean of a sample
            pixels: string
                number of points per pixel in the figures
                Recommended: 360


        Returns:
            -------
            None
        """

        self.filename = Path.cwd() / "metamodel_implementation" / filename
        self.pixels = pixels
        self.resampling_size = resampling_size

    def generate_shuffled_samples(self):
        """
        Shuffles the dataset output (phase 3) self.resampling_size times

        Parameters:
            ----------
            None

        Returns:
            -------
            all_shuffled_phase3: array of shape ((len(sample) , self.resampling_size))
                self.resampling_size times shuffled phase 3

        """
        sample = ot.Sample.ImportFromTextFile(self.filename.as_posix(), "\t", 0)
        phase3 = sample[:, -1]
        all_shuffled_phase3 = np.zeros((len(phase3), self.resampling_size))
        for i in range(self.resampling_size):
            np.random.shuffle(phase3)
            for k in range(len(phase3)):
                all_shuffled_phase3[k, i] = phase3[k, 0]
        return all_shuffled_phase3

    def generate_shuffled_samples_mechanoadaptation_elliptic(self):
        """
        Shuffles the dataset output (phase 3) self.resampling_size times

        Parameters:
            ----------
            None

        Returns:
            -------
            all_shuffled_phase3: array of shape ((len(sample) , self.resampling_size))
                self.resampling_size times shuffled phase 3

        """
        sample = ot.Sample.ImportFromTextFile(self.filename.as_posix(), "\t", 0)
        phase3 = sample[:1024, -1]
        all_shuffled_phase3 = np.zeros((len(phase3), self.resampling_size))
        for i in range(self.resampling_size):
            np.random.shuffle(phase3)
            for k in range(len(phase3)):
                all_shuffled_phase3[k, i] = phase3[k, 0]
        return all_shuffled_phase3

    def generate_shuffled_samples_mechanoadaptation_circular(self):
        """
        Shuffles the dataset output (phase 3) self.resampling_size times

        Parameters:
            ----------
            None

        Returns:
            -------
            all_shuffled_phase3: array of shape ((len(sample) , self.resampling_size))
                self.resampling_size times shuffled phase 3

        """
        sample = ot.Sample.ImportFromTextFile(self.filename.as_posix(), "\t", 0)
        phase3 = sample[:, -1]
        all_shuffled_phase3 = np.zeros((len(phase3), self.resampling_size))
        for i in range(self.resampling_size):
            np.random.shuffle(phase3)
            for k in range(len(phase3)):
                all_shuffled_phase3[k, i] = phase3[k, 0]
        return all_shuffled_phase3

    def generate_shuffled_samples_feq(self):
        """
        Shuffles the dataset output (feq) self.resampling_size times
        The input text file named filename has the following columns (all floats):
        gamma_bar_0 ; sigma_bar ; r_bar ; f_eq ; phasis number
        Parameters:
            ----------
            None

        Returns:
            -------
            all_shuffled_f_eq: array of shape ((len(sample) , self.resampling_size))
                self.resampling_size times shuffled f_eq

        """
        sample = ot.Sample.ImportFromTextFile(self.filename.as_posix(), "\t", 0)
        f_eq = sample[:, -2]
        all_shuffled_f_eq = np.zeros((len(f_eq), self.resampling_size))
        for i in range(self.resampling_size):
            np.random.shuffle(f_eq)
            for k in range(len(f_eq)):
                all_shuffled_f_eq[k, i] = f_eq[k, 0]
        return all_shuffled_f_eq

    def generate_shuffled_samples_feq_mechanoadapt_vs_passive(self):
        """
        Shuffles the dataset output (feq) self.resampling_size times
        The input text file named filename has the following columns (all floats):
        gamma_bar_0 ; sigma_bar ; gamma_bar_r ; gamma_bar_fs ; gamma_bar_lambda ; r_bar ; f_eq ; phasis number
        Parameters:
            ----------
            None

        Returns:
            -------
            all_shuffled_f_eq: array of shape ((len(sample) , self.resampling_size))
                self.resampling_size times shuffled f_eq

        """
        sample = ot.Sample.ImportFromTextFile(self.filename.as_posix(), "\t", 0)
        f_eq = sample[:, -2]
        all_shuffled_f_eq = np.zeros((len(f_eq), self.resampling_size))
        for i in range(self.resampling_size):
            np.random.shuffle(f_eq)
            for k in range(len(f_eq)):
                all_shuffled_f_eq[k, i] = f_eq[k, 0]
        return all_shuffled_f_eq

    def generate_shuffled_samples_mechanoadaptation_vs_passive_elliptic(self):
        """
        Shuffles the dataset output (feq) self.resampling_size times
        The input text file named filename has the following columns (all floats):
        gamma_bar_0 ; sigma_bar ; gamma_bar_r ; gamma_bar_fs ; gamma_bar_lambda ; r_bar ; f_eq ; phasis number
        Parameters:
            ----------
            None

        Returns:
            -------
            all_shuffled_f_eq: array of shape ((len(sample) , self.resampling_size))
                self.resampling_size times shuffled f_eq

        """
        sample = ot.Sample.ImportFromTextFile(self.filename.as_posix(), "\t", 0)
        f_eq = sample[:, -2]
        all_shuffled_f_eq = np.zeros((len(f_eq), self.resampling_size))
        for i in range(self.resampling_size):
            np.random.shuffle(f_eq)
            for k in range(len(f_eq)):
                all_shuffled_f_eq[k, i] = f_eq[k, 0]
        return all_shuffled_f_eq

    def generate_shuffled_samples_mechanoadaptation_vs_passive_circular(self):
        """
        Shuffles the dataset output (feq) self.resampling_size times
        The input text file named filename has the following columns (all floats):
        gamma_bar_0 ; sigma_bar ; gamma_bar_r ; gamma_bar_fs ; gamma_bar_lambda ; r_bar ; f_eq ; phasis number
        Parameters:
            ----------
            None

        Returns:
            -------
            all_shuffled_f_eq: array of shape ((len(sample) , self.resampling_size))
                self.resampling_size times shuffled f_eq

        """
        sample = ot.Sample.ImportFromTextFile(self.filename.as_posix(), "\t", 0)
        f_eq = sample[:, -2]
        all_shuffled_f_eq = np.zeros((len(f_eq), self.resampling_size))
        for i in range(self.resampling_size):
            np.random.shuffle(f_eq)
            for k in range(len(f_eq)):
                all_shuffled_f_eq[k, i] = f_eq[k, 0]
        return all_shuffled_f_eq

    def compute_cumulative_mean_std(self, vector):
        """
        Computes the cumulative mean and standard deviation (std) of a vector

        Parameters:
            ----------
            vector: array
                vector of which the cumulative mean and std will be computed

        Returns:
            -------
            cumulative_mean: array, same shape as vector
                cumulative mean of the vector
            cumulative_std: array, same shape as vector
                cumulative std of the vector

        """
        cumulative_mean = np.zeros(len(vector))
        cumulative_std = np.zeros_like(cumulative_mean)
        for i in range(len(vector)):
            cumulative_mean[i] = np.mean(vector[0:i])
            cumulative_std[i] = np.std(vector[0:i])
        return cumulative_mean, cumulative_std

    def compute_means_stds_of_shuffled_samples_and_export_to_pkl(self):
        """
        Computes the cumulative mean and standard deviation (std) for the
            self.resampling_size shuffled samples that
            have been generated
        Exports them into a .pkl file

        Parameters:
            ----------
            None

        Returns:
            -------
            None

        Exports:
            -------
            cumulative_mean: array
                cumulative mean of one shuffled sample
            std_of_cumulative_means: array
                std of the cumulative_mean of all the shuffled samples
            cumulative_std: array
                cumulative std of one shuffled sample
            std_of_cumulative_stds: array
                std of the cumulative_std of all the shuffled samples
            all_shuffled_phase3: array
                shuffled samples used in this method.
                output of the self.generate_shuffled_samples() method

            These objects are exported in a .pkl file named "data_representativeness.pkl"

        """
        all_shuffled_phase3 = self.generate_shuffled_samples()
        cumulative_means_for_all_samples = np.zeros_like(all_shuffled_phase3)
        cumulative_stds_for_all_samples = np.zeros_like(all_shuffled_phase3)

        std_of_cumulative_means = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        cumulative_std = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        std_of_cumulative_stds = np.zeros_like(cumulative_std)
        for i in range(self.resampling_size):
            cumulative_mean, cumulative_std = self.compute_cumulative_mean_std(all_shuffled_phase3[:, i])
            cumulative_means_for_all_samples[:, i] = cumulative_mean
            cumulative_stds_for_all_samples[:, i] = cumulative_std

        for j in range(len(std_of_cumulative_means)):
            std_of_cumulative_means[j] = np.std(cumulative_means_for_all_samples[j, :])
            std_of_cumulative_stds[j] = np.std(cumulative_stds_for_all_samples[j, :])

        with open("data_representativeness.pkl", "wb") as f:
            pickle.dump(
                [cumulative_mean, std_of_cumulative_means, cumulative_std, std_of_cumulative_stds, all_shuffled_phase3],
                f,
            )

    def compute_means_stds_of_shuffled_samples_and_export_to_pkl_constant_elliptic(self):
        """
        Computes the cumulative mean and standard deviation (std) for the
            self.resampling_size shuffled samples that
            have been generated
        Exports them into a .pkl file

        Parameters:
            ----------
            None

        Returns:
            -------
            None

        Exports:
            -------
            cumulative_mean: array
                cumulative mean of one shuffled sample
            std_of_cumulative_means: array
                std of the cumulative_mean of all the shuffled samples
            cumulative_std: array
                cumulative std of one shuffled sample
            std_of_cumulative_stds: array
                std of the cumulative_std of all the shuffled samples
            all_shuffled_phase3: array
                shuffled samples used in this method.
                output of the self.generate_shuffled_samples() method

            These objects are exported in a .pkl file named "data_representativeness.pkl"

        """
        all_shuffled_feq = self.generate_shuffled_samples_feq()
        cumulative_means_for_all_samples = np.zeros_like(all_shuffled_feq)
        cumulative_stds_for_all_samples = np.zeros_like(all_shuffled_feq)

        std_of_cumulative_means = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        cumulative_std = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        std_of_cumulative_stds = np.zeros_like(cumulative_std)
        for i in range(self.resampling_size):
            cumulative_mean, cumulative_std = self.compute_cumulative_mean_std(all_shuffled_feq[:, i])
            cumulative_means_for_all_samples[:, i] = cumulative_mean
            cumulative_stds_for_all_samples[:, i] = cumulative_std

        for j in range(len(std_of_cumulative_means)):
            std_of_cumulative_means[j] = np.std(cumulative_means_for_all_samples[j, :])
            std_of_cumulative_stds[j] = np.std(cumulative_stds_for_all_samples[j, :])

        with open("data_representativeness_constant_elliptic.pkl", "wb") as f:
            pickle.dump(
                [cumulative_mean, std_of_cumulative_means, cumulative_std, std_of_cumulative_stds, all_shuffled_feq],
                f,
            )

    def compute_means_stds_of_shuffled_samples_and_export_to_pkl_mechanoadaptation_circular(self):
        """
        Computes the cumulative mean and standard deviation (std) for the
            self.resampling_size shuffled samples that
            have been generated
        Exports them into a .pkl file

        Parameters:
            ----------
            None

        Returns:
            -------
            None

        Exports:
            -------
            cumulative_mean: array
                cumulative mean of one shuffled sample
            std_of_cumulative_means: array
                std of the cumulative_mean of all the shuffled samples
            cumulative_std: array
                cumulative std of one shuffled sample
            std_of_cumulative_stds: array
                std of the cumulative_std of all the shuffled samples
            all_shuffled_phase3: array
                shuffled samples used in this method.
                output of the self.generate_shuffled_samples() method

            These objects are exported in a .pkl file named "data_representativeness.pkl"

        """
        all_shuffled_phase3 = self.generate_shuffled_samples_mechanoadaptation_circular()
        cumulative_means_for_all_samples = np.zeros_like(all_shuffled_phase3)
        cumulative_stds_for_all_samples = np.zeros_like(all_shuffled_phase3)

        std_of_cumulative_means = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        cumulative_std = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        std_of_cumulative_stds = np.zeros_like(cumulative_std)
        for i in range(self.resampling_size):
            cumulative_mean, cumulative_std = self.compute_cumulative_mean_std(all_shuffled_phase3[:, i])
            cumulative_means_for_all_samples[:, i] = cumulative_mean
            cumulative_stds_for_all_samples[:, i] = cumulative_std

        for j in range(len(std_of_cumulative_means)):
            std_of_cumulative_means[j] = np.std(cumulative_means_for_all_samples[j, :])
            std_of_cumulative_stds[j] = np.std(cumulative_stds_for_all_samples[j, :])

        with open("data_representativeness_mechanoadaptation_circular.pkl", "wb") as f:
            pickle.dump(
                [cumulative_mean, std_of_cumulative_means, cumulative_std, std_of_cumulative_stds, all_shuffled_phase3],
                f,
            )

    def compute_means_stds_of_shuffled_samples_and_export_to_pkl_mechanoadaptation_elliptic(self):
        """
        Computes the cumulative mean and standard deviation (std) for the
            self.resampling_size shuffled samples that
            have been generated
        Exports them into a .pkl file

        Parameters:
            ----------
            None

        Returns:
            -------
            None

        Exports:
            -------
            cumulative_mean: array
                cumulative mean of one shuffled sample
            std_of_cumulative_means: array
                std of the cumulative_mean of all the shuffled samples
            cumulative_std: array
                cumulative std of one shuffled sample
            std_of_cumulative_stds: array
                std of the cumulative_std of all the shuffled samples
            all_shuffled_phase3: array
                shuffled samples used in this method.
                output of the self.generate_shuffled_samples() method

            These objects are exported in a .pkl file named "data_representativeness.pkl"

        """
        all_shuffled_phase3 = self.generate_shuffled_samples_mechanoadaptation_elliptic()
        cumulative_means_for_all_samples = np.zeros_like(all_shuffled_phase3)
        cumulative_stds_for_all_samples = np.zeros_like(all_shuffled_phase3)

        std_of_cumulative_means = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        cumulative_std = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        std_of_cumulative_stds = np.zeros_like(cumulative_std)
        for i in range(self.resampling_size):
            cumulative_mean, cumulative_std = self.compute_cumulative_mean_std(all_shuffled_phase3[:, i])
            cumulative_means_for_all_samples[:, i] = cumulative_mean
            cumulative_stds_for_all_samples[:, i] = cumulative_std

        for j in range(len(std_of_cumulative_means)):
            std_of_cumulative_means[j] = np.std(cumulative_means_for_all_samples[j, :])
            std_of_cumulative_stds[j] = np.std(cumulative_stds_for_all_samples[j, :])

        with open("data_representativeness_mechanoadaptation_elliptic.pkl", "wb") as f:
            pickle.dump(
                [cumulative_mean, std_of_cumulative_means, cumulative_std, std_of_cumulative_stds, all_shuffled_phase3],
                f,
            )

    def compute_means_stds_of_shuffled_samples_and_export_to_pkl_mechanoadaptation_vs_passive_elliptic(self):
        """
        Computes the cumulative mean and standard deviation (std) for the
            self.resampling_size shuffled samples that
            have been generated
        Exports them into a .pkl file

        Parameters:
            ----------
            None

        Returns:
            -------
            None

        Exports:
            -------
            cumulative_mean: array
                cumulative mean of one shuffled sample
            std_of_cumulative_means: array
                std of the cumulative_mean of all the shuffled samples
            cumulative_std: array
                cumulative std of one shuffled sample
            std_of_cumulative_stds: array
                std of the cumulative_std of all the shuffled samples
            all_shuffled_phase3: array
                shuffled samples used in this method.
                output of the self.generate_shuffled_samples() method

            These objects are exported in a .pkl file named "data_representativeness.pkl"

        """
        all_shuffled_feq = self.generate_shuffled_samples_mechanoadaptation_vs_passive_elliptic()
        cumulative_means_for_all_samples = np.zeros_like(all_shuffled_feq)
        cumulative_stds_for_all_samples = np.zeros_like(all_shuffled_feq)

        std_of_cumulative_means = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        cumulative_std = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        std_of_cumulative_stds = np.zeros_like(cumulative_std)
        for i in range(self.resampling_size):
            cumulative_mean, cumulative_std = self.compute_cumulative_mean_std(all_shuffled_feq[:, i])
            cumulative_means_for_all_samples[:, i] = cumulative_mean
            cumulative_stds_for_all_samples[:, i] = cumulative_std

        for j in range(len(std_of_cumulative_means)):
            std_of_cumulative_means[j] = np.std(cumulative_means_for_all_samples[j, :])
            std_of_cumulative_stds[j] = np.std(cumulative_stds_for_all_samples[j, :])

        with open("data_representativeness_mechanoadaptation_vs_passive_elliptic.pkl", "wb") as f:
            pickle.dump(
                [cumulative_mean, std_of_cumulative_means, cumulative_std, std_of_cumulative_stds, all_shuffled_feq],
                f,
            )

    def compute_means_stds_of_shuffled_samples_and_export_to_pkl_ANN_mechanoadaptation_vs_passive_elliptic(self):
        """
        Computes the cumulative mean and standard deviation (std) for the
            self.resampling_size shuffled samples that
            have been generated
        Exports them into a .pkl file

        Parameters:
            ----------
            None

        Returns:
            -------
            None

        Exports:
            -------
            cumulative_mean: array
                cumulative mean of one shuffled sample
            std_of_cumulative_means: array
                std of the cumulative_mean of all the shuffled samples
            cumulative_std: array
                cumulative std of one shuffled sample
            std_of_cumulative_stds: array
                std of the cumulative_std of all the shuffled samples
            all_shuffled_phase3: array
                shuffled samples used in this method.
                output of the self.generate_shuffled_samples() method

            These objects are exported in a .pkl file named "data_representativeness.pkl"

        """
        all_shuffled_feq = self.generate_shuffled_samples_mechanoadaptation_vs_passive_elliptic()
        cumulative_means_for_all_samples = np.zeros_like(all_shuffled_feq)
        cumulative_stds_for_all_samples = np.zeros_like(all_shuffled_feq)

        std_of_cumulative_means = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        cumulative_std = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        std_of_cumulative_stds = np.zeros_like(cumulative_std)
        for i in range(self.resampling_size):
            cumulative_mean, cumulative_std = self.compute_cumulative_mean_std(all_shuffled_feq[:, i])
            cumulative_means_for_all_samples[:, i] = cumulative_mean
            cumulative_stds_for_all_samples[:, i] = cumulative_std

        for j in range(len(std_of_cumulative_means)):
            std_of_cumulative_means[j] = np.std(cumulative_means_for_all_samples[j, :])
            std_of_cumulative_stds[j] = np.std(cumulative_stds_for_all_samples[j, :])

        with open("data_representativeness_ANN_mechanoadaptation_vs_passive_elliptic.pkl", "wb") as f:
            pickle.dump(
                [cumulative_mean, std_of_cumulative_means, cumulative_std, std_of_cumulative_stds, all_shuffled_feq],
                f,
            )

    def compute_means_stds_of_shuffled_samples_and_export_to_pkl_mechanoadaptation_vs_passive_circular(self):
        """
        Computes the cumulative mean and standard deviation (std) for the
            self.resampling_size shuffled samples that
            have been generated
        Exports them into a .pkl file

        Parameters:
            ----------
            None

        Returns:
            -------
            None

        Exports:
            -------
            cumulative_mean: array
                cumulative mean of one shuffled sample
            std_of_cumulative_means: array
                std of the cumulative_mean of all the shuffled samples
            cumulative_std: array
                cumulative std of one shuffled sample
            std_of_cumulative_stds: array
                std of the cumulative_std of all the shuffled samples
            all_shuffled_phase3: array
                shuffled samples used in this method.
                output of the self.generate_shuffled_samples() method

            These objects are exported in a .pkl file named "data_representativeness.pkl"

        """
        all_shuffled_feq = self.generate_shuffled_samples_mechanoadaptation_vs_passive_circular()
        cumulative_means_for_all_samples = np.zeros_like(all_shuffled_feq)
        cumulative_stds_for_all_samples = np.zeros_like(all_shuffled_feq)

        std_of_cumulative_means = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        cumulative_std = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        std_of_cumulative_stds = np.zeros_like(cumulative_std)
        for i in range(self.resampling_size):
            cumulative_mean, cumulative_std = self.compute_cumulative_mean_std(all_shuffled_feq[:, i])
            cumulative_means_for_all_samples[:, i] = cumulative_mean
            cumulative_stds_for_all_samples[:, i] = cumulative_std

        for j in range(len(std_of_cumulative_means)):
            std_of_cumulative_means[j] = np.std(cumulative_means_for_all_samples[j, :])
            std_of_cumulative_stds[j] = np.std(cumulative_stds_for_all_samples[j, :])

        with open("data_representativeness_mechanoadaptation_vs_passive_circular.pkl", "wb") as f:
            pickle.dump(
                [cumulative_mean, std_of_cumulative_means, cumulative_std, std_of_cumulative_stds, all_shuffled_feq],
                f,
            )

    def compute_means_stds_of_shuffled_samples_and_export_to_pkl_feq(self):
        """
        Computes the cumulative mean and standard deviation (std) for the
            self.resampling_size shuffled samples that
            have been generated
        Exports them into a .pkl file

        Parameters:
            ----------
            None

        Returns:
            -------
            None

        Exports:
            -------
            cumulative_mean: array
                cumulative mean of one shuffled sample
            std_of_cumulative_means: array
                std of the cumulative_mean of all the shuffled samples
            cumulative_std: array
                cumulative std of one shuffled sample
            std_of_cumulative_stds: array
                std of the cumulative_std of all the shuffled samples
            all_shuffled_f_eq: array
                shuffled samples used in this method.
                output of the self.generate_shuffled_samples() method

            These objects are exported in a .pkl file named "data_representativeness.pkl"

        """
        all_shuffled_f_eq = self.generate_shuffled_samples_feq()
        cumulative_means_for_all_samples = np.zeros_like(all_shuffled_f_eq)
        cumulative_stds_for_all_samples = np.zeros_like(all_shuffled_f_eq)

        std_of_cumulative_means = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        cumulative_std = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        std_of_cumulative_stds = np.zeros_like(cumulative_std)
        for i in range(self.resampling_size):
            cumulative_mean, cumulative_std = self.compute_cumulative_mean_std(all_shuffled_f_eq[:, i])
            cumulative_means_for_all_samples[:, i] = cumulative_mean
            cumulative_stds_for_all_samples[:, i] = cumulative_std

        for j in range(len(std_of_cumulative_means)):
            std_of_cumulative_means[j] = np.std(cumulative_means_for_all_samples[j, :])
            std_of_cumulative_stds[j] = np.std(cumulative_stds_for_all_samples[j, :])

        with open("data_representativeness_feq.pkl", "wb") as f:
            pickle.dump(
                [cumulative_mean, std_of_cumulative_means, cumulative_std, std_of_cumulative_stds, all_shuffled_f_eq],
                f,
            )

    def plot_cumulative_mean_vs_sample_size(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the cumulative mean of a sample with the
            std (computed from the self.resampling_size shuffled
                samples)

        Parameters:
            ----------
            None

        Returns:
            -------
            None

        """
        with open("data_representativeness.pkl", "rb") as f:
            [cumulative_mean, std_of_cumulative_means, _, _, all_shuffled_phase3] = pickle.load(f)

        sample_size = np.arange(1, len(cumulative_mean) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        ax.errorbar(
            sample_size,
            self.compute_cumulative_mean_std(all_shuffled_phase3[:, 0])[0],
            std_of_cumulative_means,
            color="black",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax.set_xticks([1, 250, 500, 750, 1000])
        ax.set_xticklabels(
            [1, 250, 500, 750, 1000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(-0.02, 0.52)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_yticklabels(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Mean of $\Psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "cumulative_mean_vs_sample_size")

    def plot_cumulative_mean_vs_sample_size_constant_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the cumulative mean of a sample with the
            std (computed from the self.resampling_size shuffled
                samples)

        Parameters:
            ----------
            None

        Returns:
            -------
            None

        """
        with open("data_representativeness_constant_elliptic.pkl", "rb") as f:
            [cumulative_mean, std_of_cumulative_means, _, _, all_shuffled_feq] = pickle.load(f)

        sample_size = np.arange(1, len(cumulative_mean) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        cum_mean = self.compute_cumulative_mean_std(all_shuffled_feq[:, 0])
        ax.plot(
            sample_size,
            cum_mean[0],
            # std_of_cumulative_means,
            color="black",
            linewidth=1,
            # ecolor="gray",
            # elinewidth=0.5,
        )
        ax.fill_between(sample_size, cum_mean[0] + std_of_cumulative_means, cum_mean[0] - std_of_cumulative_means, color='k', alpha = 0.2)
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(0.24, 0.36)
        ax.set_yticks([0.25, 0.3, 0.35])
        ax.set_yticklabels(
            [0.25, 0.3, 0.35],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Mean of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "cumulative_mean_vs_sample_size_constant_elliptic")

    def plot_cumulative_mean_vs_sample_size_mechanoadaptation_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the cumulative mean of a sample with the std 
            (computed from the self.resampling_size shuffled samples)
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_elliptic.pkl", "rb") as f:
            [cumulative_mean, std_of_cumulative_means, _, _, all_shuffled_psi3] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_mean) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        ax.errorbar(
            sample_size,
            self.compute_cumulative_mean_std(all_shuffled_psi3[:, 0])[0],
            std_of_cumulative_means,
            color="black",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax.set_xticks([1, 200, 400, 600, 800, 1000])
        ax.set_xticklabels(
            [1, 200, 400, 600, 800, 1000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(0.05, 0.2)
        ax.set_yticks([0.05, 0.1, 0.15, 0.2])
        ax.set_yticklabels(
            [0.05, 0.1, 0.15, 0.2],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Mean of $\Psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "cumulative_mean_vs_sample_size_mechanoadaptation_elliptic")

    def plot_cumulative_mean_vs_sample_size_mechanoadaptation_vs_passive_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the cumulative mean of a sample with the std 
            (computed from the self.resampling_size shuffled samples)
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_vs_passive_elliptic.pkl", "rb") as f:
            [cumulative_mean, std_of_cumulative_means, _, _, all_shuffled_feq] = pickle.load(f)

        sample_size = np.arange(1, len(cumulative_mean) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        ax.errorbar(
            sample_size,
            self.compute_cumulative_mean_std(all_shuffled_feq[:, 0])[0],
            std_of_cumulative_means,
            color="black",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(0.23, 0.47)
        ax.set_yticks([0.3, 0.4])
        ax.set_yticklabels(
            [0.3, 0.4],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Mean of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "cumulative_mean_vs_sample_size_mechanoadaptation_vs_passive_elliptic")

    def plot_cumulative_mean_vs_sample_size_ANN_mechanoadaptation_vs_passive_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the cumulative mean of a sample with the std 
            (computed from the self.resampling_size shuffled samples)
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_ANN_mechanoadaptation_vs_passive_elliptic.pkl", "rb") as f:
            [cumulative_mean, std_of_cumulative_means, _, _, all_shuffled_feq] = pickle.load(f)

        sample_size = np.arange(1, len(cumulative_mean) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        ax.errorbar(
            sample_size,
            self.compute_cumulative_mean_std(all_shuffled_feq[:, 0])[0],
            std_of_cumulative_means,
            color="black",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(0.23, 0.47)
        ax.set_yticks([0.3, 0.4])
        ax.set_yticklabels(
            [0.3, 0.4],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Mean of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "cumulative_mean_vs_sample_size_ANN_mechanoadaptation_vs_passive_elliptic")


    def plot_cumulative_std_vs_sample_size_constant_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the cumulative std of a sample with the std 
            (computed from the self.resampling_size shuffled samples)
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_constant_elliptic.pkl", "rb") as f:
            [_, _, cumulative_std, std_of_cumulative_stds, all_shuffled_feq] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        ax.errorbar(
            sample_size,
            self.compute_cumulative_mean_std(all_shuffled_feq[:, 0])[1],
            std_of_cumulative_stds,
            color="black",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(0.19, 0.26)
        ax.set_yticks([0.2, 0.225, 0.25])
        ax.set_yticklabels(
            [0.2, 0.225, 0.25],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Std of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "cumulative_std_vs_sample_size_constant_elliptic")

    def plot_cumulative_std_vs_sample_size_mechanoadaptation_circular(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the cumulative std of a sample with the std 
            (computed from the self.resampling_size shuffled samples)
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_circular.pkl", "rb") as f:
            [_, _, cumulative_std, std_of_cumulative_stds, all_shuffled_psi3] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        ax.errorbar(
            sample_size,
            self.compute_cumulative_mean_std(all_shuffled_psi3[:, 0])[1],
            std_of_cumulative_stds,
            color="black",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax.set_xticks([1, 200, 400, 600, 800, 1000])
        ax.set_xticklabels(
            [1, 200, 400, 600, 800, 1000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(0.1, 0.2)
        ax.set_yticks([0.1, 0.15, 0.2])
        ax.set_yticklabels(
            [0.1, 0.15, 0.2],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Std of $\Psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "cumulative_std_vs_sample_size_mechanoadaptation_circular")

    def plot_cumulative_std_vs_sample_size_mechanoadaptation_vs_passive_circular(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the cumulative std of a sample with the std 
            (computed from the self.resampling_size shuffled samples)
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_vs_passive_circular.pkl", "rb") as f:
            [_, _, cumulative_std, std_of_cumulative_stds, all_shuffled_feq] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        ax.errorbar(
            sample_size,
            self.compute_cumulative_mean_std(all_shuffled_feq[:, 0])[1],
            std_of_cumulative_stds,
            color="black",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(0.28, 0.42)
        ax.set_yticks([0.3, 0.35, 0.4])
        ax.set_yticklabels(
            [0.3, 0.35, 0.4],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Std of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "cumulative_std_vs_sample_size_mechanoadaptation_vs_passive_circular")

    def plot_cumulative_std_vs_sample_size_mechanoadaptation_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the cumulative std of a sample with the std 
            (computed from the self.resampling_size shuffled samples)
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_elliptic.pkl", "rb") as f:
            [_, _, cumulative_std, std_of_cumulative_stds, all_shuffled_psi3] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        ax.errorbar(
            sample_size,
            self.compute_cumulative_mean_std(all_shuffled_psi3[:, 0])[1],
            std_of_cumulative_stds,
            color="black",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax.set_xticks([1, 200, 400, 600, 800, 1000])
        ax.set_xticklabels(
            [1, 200, 400, 600, 800, 1000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(0.15, 0.25)
        ax.set_yticks([0.15, 0.2, 0.25])
        ax.set_yticklabels(
            [0.15, 0.2, 0.25],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Std of $\Psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "cumulative_std_vs_sample_size_mechanoadaptation_elliptic")

    def plot_cumulative_std_vs_sample_size_mechanoadaptation_vs_passive_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the cumulative std of a sample with the std 
            (computed from the self.resampling_size shuffled samples)
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_vs_passive_elliptic.pkl", "rb") as f:
            [_, _, cumulative_std, std_of_cumulative_stds, all_shuffled_feq] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        ax.errorbar(
            sample_size,
            self.compute_cumulative_mean_std(all_shuffled_feq[:, 0])[1],
            std_of_cumulative_stds,
            color="black",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(0.23, 0.37)
        ax.set_yticks([0.25, 0.3, 0.35])
        ax.set_yticklabels(
            [0.25, 0.3, 0.35],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Std of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "cumulative_std_vs_sample_size_mechanoadaptation_vs_passive_elliptic")


    def plot_cumulative_std_vs_sample_size_ANN_mechanoadaptation_vs_passive_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the cumulative std of a sample with the std 
            (computed from the self.resampling_size shuffled samples)
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_ANN_mechanoadaptation_vs_passive_elliptic.pkl", "rb") as f:
            [_, _, cumulative_std, std_of_cumulative_stds, all_shuffled_feq] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        ax.errorbar(
            sample_size,
            self.compute_cumulative_mean_std(all_shuffled_feq[:, 0])[1],
            std_of_cumulative_stds,
            color="black",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(0.23, 0.37)
        ax.set_yticks([0.25, 0.3, 0.35])
        ax.set_yticklabels(
            [0.25, 0.3, 0.35],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Std of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "cumulative_std_vs_sample_size_ANN_mechanoadaptation_vs_passive_elliptic")


    def plot_gradient_cumulative_mean_vs_sample_size_constant_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute normalized gradient of the cumulative mean of a sample
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_constant_elliptic.pkl", "rb") as f:
            [mean_of_cumulative_means, _, _, _, _] = pickle.load(f)
        sample_size = np.arange(1, len(mean_of_cumulative_means) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [
            np.abs(np.diff(mean_of_cumulative_means)[k]) / mean_of_cumulative_means[k]
            for k in range(len(mean_of_cumulative_means) - 1)
        ]
        ax.plot(sample_size[0:-1], gradient, "-k")
        ax.plot(sample_size[0:-1], [1e-2] * len(sample_size[0:-1]), "--r")
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 5e-1)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Mean of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "gradient_cumulative_mean_vs_sample_size_constant_elliptic")

    def plot_gradient_cumulative_mean_vs_sample_size_mechanoadaptation_circular(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute normalized gradient of the cumulative mean of a sample
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_circular.pkl", "rb") as f:
            [mean_of_cumulative_means, _, _, _, _] = pickle.load(f)
        sample_size = np.arange(1, len(mean_of_cumulative_means) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [
            np.abs(np.diff(mean_of_cumulative_means)[k]) / mean_of_cumulative_means[k]
            for k in range(len(mean_of_cumulative_means) - 1)
        ]
        ax.plot(sample_size[0:-1], gradient, "-k")
        ax.plot(sample_size[0:-1], [1e-2] * len(sample_size[0:-1]), "--r")
        ax.set_xticks([1, 200, 400, 600, 800, 1000])
        ax.set_xticklabels(
            [1, 200, 400, 600, 800, 1000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-6, 5e-1)
        ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Mean of $\Psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "gradient_cumulative_mean_vs_sample_size_mechanoadaptation_circular")

    def plot_gradient_cumulative_mean_vs_sample_size_mechanoadaptation_vs_passive_circular(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute normalized gradient of the cumulative mean of a sample
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_vs_passive_circular.pkl", "rb") as f:
            [mean_of_cumulative_means, _, _, _, _] = pickle.load(f)
        sample_size = np.arange(1, len(mean_of_cumulative_means) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [
            np.abs(np.diff(mean_of_cumulative_means)[k]) / mean_of_cumulative_means[k]
            for k in range(len(mean_of_cumulative_means) - 1)
        ]
        ax.plot(sample_size[0:-1], gradient, "-k")
        ax.plot(sample_size[0:-1], [1e-2] * len(sample_size[0:-1]), "--r")
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 5e-1)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Mean of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "gradient_cumulative_mean_vs_sample_size_mechanoadaptation_vs_passive_circular")

    def plot_gradient_cumulative_mean_vs_sample_size_mechanoadaptation_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute normalized gradient of the cumulative mean of a sample
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_elliptic.pkl", "rb") as f:
            [mean_of_cumulative_means, _, _, _, _] = pickle.load(f)
        sample_size = np.arange(1, len(mean_of_cumulative_means) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [
            np.abs(np.diff(mean_of_cumulative_means)[k]) / mean_of_cumulative_means[k]
            for k in range(len(mean_of_cumulative_means) - 1)
        ]
        ax.plot(sample_size[0:-1], gradient, "-k")
        ax.plot(sample_size[0:-1], [1e-2] * len(sample_size[0:-1]), "--r")
        ax.set_xticks([1, 200, 400, 600, 800, 1000])
        ax.set_xticklabels(
            [1, 200, 400, 600, 800, 1000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-6, 5e-1)
        ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Mean of $\Psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "gradient_cumulative_mean_vs_sample_size_mechanoadaptation_elliptic")

    def plot_gradient_cumulative_mean_vs_sample_size_mechanoadaptation_vs_passive_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute normalized gradient of the cumulative mean of a sample
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_vs_passive_elliptic.pkl", "rb") as f:
            [mean_of_cumulative_means, _, _, _, _] = pickle.load(f)
        sample_size = np.arange(1, len(mean_of_cumulative_means) + 1, 2)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [
            np.abs(np.diff(mean_of_cumulative_means)[k]) / mean_of_cumulative_means[k]
            for k in range(0, len(mean_of_cumulative_means) -1, 2)
        ]
        ax.plot(sample_size[0:-1], gradient[0:-1], "-k")
        ax.plot(sample_size[0:-1], [1e-2] * len(sample_size[0:-1]), "--r")
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 5e-1)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Mean of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "gradient_cumulative_mean_vs_sample_size_mechanoadaptation_vs_passive_elliptic")


    def plot_gradient_cumulative_mean_vs_sample_size_ANN_mechanoadaptation_vs_passive_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute normalized gradient of the cumulative mean of a sample
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_ANN_mechanoadaptation_vs_passive_elliptic.pkl", "rb") as f:
            [mean_of_cumulative_means, _, _, _, _] = pickle.load(f)
        sample_size = np.arange(1, len(mean_of_cumulative_means) + 1, 2)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [
            np.abs(np.diff(mean_of_cumulative_means)[k]) / mean_of_cumulative_means[k]
            for k in range(0, len(mean_of_cumulative_means) -1, 2)
        ]
        ax.plot(sample_size[0:-1], gradient[0:-1], "-k")
        ax.plot(sample_size[0:-1], [1e-2] * len(sample_size[0:-1]), "--r")
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 5e-1)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Mean of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "gradient_cumulative_mean_vs_sample_size_ANN_mechanoadaptation_vs_passive_elliptic")


    def plot_gradient_cumulative_std_vs_size_constant_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute normalized gradient of the cumulative std of a sample
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_constant_elliptic.pkl", "rb") as f:
            [_, _, cumulative_std, _, _] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [np.abs(np.diff(cumulative_std)[k]) / cumulative_std[k] for k in range(2, len(cumulative_std) - 1)]
        ax.plot(sample_size[2:-1], gradient, "-k")
        ax.plot(sample_size[2:-1], [1e-2] * len(sample_size[2:-1]), "--r")
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 5e-1)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Std of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "gradient_cumulative_std_vs_sample_size_constant_elliptic")

    def plot_gradient_cumulative_std_vs_size_mechanoadaptation_circular(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute normalized gradient of the cumulative std of a sample
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_circular.pkl", "rb") as f:
            [_, _, cumulative_std, _, _] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [np.abs(np.diff(cumulative_std)[k]) / cumulative_std[k] for k in range(2, len(cumulative_std) - 1)]
        ax.plot(sample_size[2:-1], gradient, "-k")
        ax.plot(sample_size[2:-1], [1e-2] * len(sample_size[2:-1]), "--r")
        ax.set_xticks([1, 200, 400, 600, 800, 1000])
        ax.set_xticklabels(
            [1, 200, 400, 600, 800, 1000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-6, 5e-1)
        ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Std of $\Psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "gradient_cumulative_std_vs_sample_size_mechanoadaptation_circular")

    def plot_gradient_cumulative_std_vs_size_mechanoadaptation_vs_passive_circular(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute normalized gradient of the cumulative std of a sample
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_vs_passive_circular.pkl", "rb") as f:
            [_, _, cumulative_std, _, _] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [np.abs(np.diff(cumulative_std)[k]) / cumulative_std[k] for k in range(2, len(cumulative_std) - 1)]
        ax.plot(sample_size[2:-1], gradient, "-k")
        ax.plot(sample_size[2:-1], [1e-2] * len(sample_size[2:-1]), "--r")
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 5e-1)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Std of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "gradient_cumulative_std_vs_sample_size_mechanoadaptation_vs_passive_circular")

    def plot_gradient_cumulative_std_vs_size_mechanoadaptation_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute normalized gradient of the cumulative std of a sample
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_elliptic.pkl", "rb") as f:
            [_, _, cumulative_std, _, _] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [np.abs(np.diff(cumulative_std)[k]) / cumulative_std[k] for k in range(2, len(cumulative_std) - 1)]
        ax.plot(sample_size[2:-1], gradient, "-k")
        ax.plot(sample_size[2:-1], [1e-2] * len(sample_size[2:-1]), "--r")
        ax.set_xticks([1, 200, 400, 600, 800, 1000])
        ax.set_xticklabels(
            [1, 200, 400, 600, 800, 1000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-6, 5e-1)
        ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Std of $\Psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "gradient_cumulative_std_vs_sample_size_mechanoadaptation_elliptic")

    def plot_gradient_cumulative_std_vs_size_mechanoadaptation_vs_passive_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute normalized gradient of the cumulative std of a sample
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_vs_passive_elliptic.pkl", "rb") as f:
            [_, _, cumulative_std, _, _] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [np.abs(np.diff(cumulative_std)[k]) / cumulative_std[k] for k in range(2, len(cumulative_std) - 1)]
        ax.plot(sample_size[2:-1], gradient, "-k")
        ax.plot(sample_size[2:-1], [1e-2] * len(sample_size[2:-1]), "--r")
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 5e-1)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Std of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "gradient_cumulative_std_vs_sample_size_mechanoadaptation_vs_passive_elliptic")

    def plot_gradient_cumulative_std_vs_size_ANN_mechanoadaptation_vs_passive_elliptic(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute normalized gradient of the cumulative std of a sample
        Parameters:
            ----------
            None
        Returns:
            -------
            None
        """
        with open("data_representativeness_ANN_mechanoadaptation_vs_passive_elliptic.pkl", "rb") as f:
            [_, _, cumulative_std, _, _] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [np.abs(np.diff(cumulative_std)[k]) / cumulative_std[k] for k in range(2, len(cumulative_std) - 1)]
        ax.plot(sample_size[2:-1], gradient, "-k")
        ax.plot(sample_size[2:-1], [1e-2] * len(sample_size[2:-1]), "--r")
        ax.set_xticks([1, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(
            [1, 1000, 2000, 3000, 4000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 5e-1)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Std of $\tilde{F}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid(linestyle='--')
        savefigure.save_as_png(fig, "gradient_cumulative_std_vs_sample_size_ANN_mechanoadaptation_vs_passive_elliptic")


    def plot_PDF_sample_constant_elliptic(self, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
        """
        Plots the histogram and an approximation of the PCE of the data
        Parameters:
            ----------
            nb_bin: float
                number of bins used to build the histogram
        Returns:
            -------
            None
        """
        with open("data_representativeness_constant_elliptic.pkl", "rb") as f:
            [_, _, _, _, all_shuffled_feq] = pickle.load(f)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        X_plot = np.linspace(0, 1, 2000)[:, None]
        kde_model = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(all_shuffled_feq[:, 0].reshape(-1, 1))
        log_dens_model = kde_model.score_samples(X_plot)
        ax.hist(all_shuffled_feq[:, 0], bins=nb_bin, density=True, color="lightgray", ec="black")
        ax.plot(
            X_plot[:, 0],
            np.exp(log_dens_model),
            color='k',
            lw=2,
            linestyle="-", label="model",
        )
        ax.set_xticks(xticks.energy_plots())
        ax.set_xticklabels(
            xticklabels.energy_plots(),
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
        savefigure.save_as_png(fig, "PDF_feq_constant_elliptic" + str(pixels))

    def plot_PDF_sample_mechanoadaptation_circular(self, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
        """
        Plots the histogram and an approximation of the PCE of the data
        Parameters:
            ----------
            nb_bin: float
                number of bins used to build the histogram
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_circular.pkl", "rb") as f:
            [_, _, _, _, all_shuffled_phase3] = pickle.load(f)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        X_plot = np.linspace(0, 1, 1000)[:, None]
        kde_model = KernelDensity(kernel='gaussian', bandwidth=0.03).fit(all_shuffled_phase3[:, 0].reshape(-1, 1))
        log_dens_model = kde_model.score_samples(X_plot)
        ax.hist(all_shuffled_phase3[:, 0], bins=nb_bin, density=True, color="lightgray", ec="black")
        ax.plot(
            X_plot[:, 0],
            np.exp(log_dens_model),
            color='k',
            lw=2,
            linestyle="-", label="model",
        )
        ax.set_xticks(xticks.energy_plots())
        ax.set_xticklabels(
            xticklabels.energy_plots(),
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
        ax.set_xlabel(r"$\psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel("$p_{\Psi_3}(\psi_3)$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        savefigure.save_as_png(fig, "PDF_psi3_mechanoadaptation_circular" + str(pixels))

    def plot_PDF_sample_mechanoadaptation_vs_passive_circular(self, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
        """
        Plots the histogram and an approximation of the PCE of the data
        Parameters:
            ----------
            nb_bin: float
                number of bins used to build the histogram
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_vs_passive_circular.pkl", "rb") as f:
            [_, _, _, _, all_shuffled_feq] = pickle.load(f)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        X_plot = np.linspace(0, 1, 2000)[:, None]
        kde_model = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(all_shuffled_feq[:, 0].reshape(-1, 1))
        log_dens_model = kde_model.score_samples(X_plot)
        ax.hist(all_shuffled_feq[:, 0], bins=nb_bin, density=True, color="lightgray", ec="black")
        ax.plot(
            X_plot[:, 0],
            np.exp(log_dens_model),
            color='k',
            lw=2,
            linestyle="-", label="model",
        )
        ax.set_xticks(xticks.energy_plots())
        ax.set_xticklabels(
            xticklabels.energy_plots(),
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
        savefigure.save_as_png(fig, "PDF_feq_mechanoadaptation_vs_passive_circular" + str(pixels))

    def plot_PDF_sample_mechanoadaptation_elliptic(self, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
        """
        Plots the histogram and an approximation of the PCE of the data
        Parameters:
            ----------
            nb_bin: float
                number of bins used to build the histogram
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_elliptic.pkl", "rb") as f:
            [_, _, _, _, all_shuffled_phase3] = pickle.load(f)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        X_plot = np.linspace(0, 1, 1000)[:, None]
        kde_model = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(all_shuffled_phase3[:, 0].reshape(-1, 1))
        log_dens_model = kde_model.score_samples(X_plot)
        ax.hist(all_shuffled_phase3[:, 0], bins=nb_bin, density=True, color="lightgray", ec="black")
        ax.plot(
            X_plot[:, 0],
            np.exp(log_dens_model),
            color='k',
            lw=2,
            linestyle="-", label="model",
        )
        ax.set_xticks(xticks.energy_plots())
        ax.set_xticklabels(
            xticklabels.energy_plots(),
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim((0, 13))
        ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
        ax.set_yticklabels(
            [0, 2, 4, 6, 8, 10, 12],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlim((-0.02, 1.02))
        ax.set_xlabel(r"$\Psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel("$p_{\Psi_3}(\psi_3)$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        savefigure.save_as_png(fig, "PDF_psi3_mechanoadaptation_elliptic" + str(pixels))

    def plot_PDF_sample_mechanoadaptation_vs_passive_elliptic(self, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
        """
        Plots the histogram and an approximation of the PCE of the data
        Parameters:
            ----------
            nb_bin: float
                number of bins used to build the histogram
        Returns:
            -------
            None
        """
        with open("data_representativeness_mechanoadaptation_vs_passive_elliptic.pkl", "rb") as f:
            [_, _, _, _, all_shuffled_feq] = pickle.load(f)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        X_plot = np.linspace(0, 1, 2000)[:, None]
        kde_model = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(all_shuffled_feq[:, 0].reshape(-1, 1))
        log_dens_model = kde_model.score_samples(X_plot)
        ax.hist(all_shuffled_feq[:, 0], bins=nb_bin, density=True, color="lightgray", ec="black")
        ax.plot(
            X_plot[:, 0],
            np.exp(log_dens_model),
            color='k',
            lw=2,
            linestyle="-", label="model",
        )
        ax.set_xticks(xticks.energy_plots())
        ax.set_xticklabels(
            xticklabels.energy_plots(),
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
        savefigure.save_as_png(fig, "PDF_feq_mechanoadaptation_vs_passive_elliptic" + str(pixels))

    def plot_PDF_sample_ANN_mechanoadaptation_vs_passive_elliptic(self, nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels):
        """
        Plots the histogram and an approximation of the PCE of the data
        Parameters:
            ----------
            nb_bin: float
                number of bins used to build the histogram
        Returns:
            -------
            None
        """
        with open("data_representativeness_ANN_mechanoadaptation_vs_passive_elliptic.pkl", "rb") as f:
            [_, _, _, _, all_shuffled_feq] = pickle.load(f)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        X_plot = np.linspace(0, 1, 2000)[:, None]
        kde_model = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(all_shuffled_feq[:, 0].reshape(-1, 1))
        log_dens_model = kde_model.score_samples(X_plot)
        ax.hist(all_shuffled_feq[:, 0], bins=nb_bin, density=False, color="lightgray", ec="black")
        ax.plot(
            X_plot[:, 0],
            np.exp(log_dens_model),
            color='k',
            lw=2,
            linestyle="-", label="model",
        )
        ax.set_xticks(xticks.energy_plots())
        ax.set_xticklabels(
            xticklabels.energy_plots(),
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        # ax.set_yticks([0, 2, 4, 6])
        # ax.set_yticklabels(
        #     [0, 2, 4, 6],
        #     font=fonts.serif(),
        #     fontsize=fonts.axis_legend_size(),
        # )
        ax.set_xlim((-0.02, 1.02))
        # ax.set_ylim((0, 7))
        ax.set_xlabel(r"$\tilde{f}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"$p_{\tilde{F}}(\tilde{f})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        savefigure.save_as_png(fig, "PDF_feq_ANN_mechanoadaptation_vs_passive_elliptic" + str(pixels))

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    xticks = XTicks()
    xticklabels = XTickLabels()
    nb_bin = 20

    # ### Constant elliptic ### 

    # filename_qMC_Sobol_constant_elliptic = "dataset_for_metamodel_creation_feq_constant_elliptic.txt"

    # samplerepresentativeness_constant_elliptic = SampleRepresentativeness(filename_qMC_Sobol_constant_elliptic, resampling_size=200, pixels=360)

    # samplerepresentativeness_constant_elliptic.compute_means_stds_of_shuffled_samples_and_export_to_pkl_constant_elliptic()

    # samplerepresentativeness_constant_elliptic.plot_cumulative_mean_vs_sample_size_constant_elliptic(createfigure, savefigure, fonts)

    # samplerepresentativeness_constant_elliptic.plot_cumulative_std_vs_sample_size_constant_elliptic(createfigure, savefigure, fonts)

    # samplerepresentativeness_constant_elliptic.plot_gradient_cumulative_mean_vs_sample_size_constant_elliptic(createfigure, savefigure, fonts)

    # samplerepresentativeness_constant_elliptic.plot_gradient_cumulative_std_vs_size_constant_elliptic(createfigure, savefigure, fonts)

    # samplerepresentativeness_constant_elliptic.plot_PDF_sample_constant_elliptic(
    #     nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360
    # )

    # ### Mechano-adaptation circular ### 

    # filename_qMC_Sobol_mechanoadaptation_circular = "dataset_for_metamodel_creation_mechanoadaptation_circular.txt"

    # samplerepresentativeness_mechanoadaptation_circular = SampleRepresentativeness(
    #     filename_qMC_Sobol_mechanoadaptation_circular, resampling_size=200, pixels=360)

    # samplerepresentativeness_mechanoadaptation_circular.compute_means_stds_of_shuffled_samples_and_export_to_pkl_mechanoadaptation_circular()

    # samplerepresentativeness_mechanoadaptation_circular.plot_cumulative_mean_vs_sample_size_mechanoadaptation_circular(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_circular.plot_cumulative_std_vs_sample_size_mechanoadaptation_circular(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_circular.plot_gradient_cumulative_mean_vs_sample_size_mechanoadaptation_circular(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_circular.plot_gradient_cumulative_std_vs_size_mechanoadaptation_circular(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_circular.plot_PDF_sample_mechanoadaptation_circular(
    #     nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360
    # )

    # ### Mechano-adaptation vs passive circular ### 

    # filename_qMC_Sobol_mechanoadaptation_vs_passive_circular = "dataset_for_metamodel_creation_mechanoadaptation_vs_passive_circular.txt"

    # samplerepresentativeness_mechanoadaptation_vs_passive_circular = SampleRepresentativeness(
    #     filename_qMC_Sobol_mechanoadaptation_vs_passive_circular, resampling_size=200, pixels=360)

    # samplerepresentativeness_mechanoadaptation_vs_passive_circular.compute_means_stds_of_shuffled_samples_and_export_to_pkl_mechanoadaptation_vs_passive_circular()

    # samplerepresentativeness_mechanoadaptation_vs_passive_circular.plot_cumulative_mean_vs_sample_size_mechanoadaptation_vs_passive_circular(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_vs_passive_circular.plot_cumulative_std_vs_sample_size_mechanoadaptation_vs_passive_circular(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_vs_passive_circular.plot_gradient_cumulative_mean_vs_sample_size_mechanoadaptation_vs_passive_circular(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_vs_passive_circular.plot_gradient_cumulative_std_vs_size_mechanoadaptation_vs_passive_circular(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_vs_passive_circular.plot_PDF_sample_mechanoadaptation_vs_passive_circular(
    #     nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360
    # )

    # ### Mechano-adaptation elliptic ### 

    # filename_qMC_Sobol_mechanoadaptation_elliptic = "dataset_for_metamodel_creation_mechanoadaptation_elliptic.txt"

    # samplerepresentativeness_mechanoadaptation_elliptic = SampleRepresentativeness(
    #     filename_qMC_Sobol_mechanoadaptation_elliptic, resampling_size=200, pixels=360)

    # samplerepresentativeness_mechanoadaptation_elliptic.compute_means_stds_of_shuffled_samples_and_export_to_pkl_mechanoadaptation_elliptic()

    # samplerepresentativeness_mechanoadaptation_elliptic.plot_cumulative_mean_vs_sample_size_mechanoadaptation_elliptic(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_elliptic.plot_cumulative_std_vs_sample_size_mechanoadaptation_elliptic(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_elliptic.plot_gradient_cumulative_mean_vs_sample_size_mechanoadaptation_elliptic(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_elliptic.plot_gradient_cumulative_std_vs_size_mechanoadaptation_elliptic(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_elliptic.plot_PDF_sample_mechanoadaptation_elliptic(
    #     nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360
    # )


    # ### Mechano-adaptation vs passive elliptic ### 

    # filename_qMC_Sobol_mechanoadaptation_vs_passive_elliptic = "dataset_for_metamodel_creation_mechanoadaptation_vs_passive_elliptic.txt"

    # samplerepresentativeness_mechanoadaptation_vs_passive_elliptic = SampleRepresentativeness(
    #     filename_qMC_Sobol_mechanoadaptation_vs_passive_elliptic, resampling_size=200, pixels=360)

    # samplerepresentativeness_mechanoadaptation_vs_passive_elliptic.compute_means_stds_of_shuffled_samples_and_export_to_pkl_mechanoadaptation_vs_passive_elliptic()

    # samplerepresentativeness_mechanoadaptation_vs_passive_elliptic.plot_cumulative_mean_vs_sample_size_mechanoadaptation_vs_passive_elliptic(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_vs_passive_elliptic.plot_cumulative_std_vs_sample_size_mechanoadaptation_vs_passive_elliptic(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_vs_passive_elliptic.plot_gradient_cumulative_mean_vs_sample_size_mechanoadaptation_vs_passive_elliptic(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_vs_passive_elliptic.plot_gradient_cumulative_std_vs_size_mechanoadaptation_vs_passive_elliptic(createfigure, savefigure, fonts)

    # samplerepresentativeness_mechanoadaptation_vs_passive_elliptic.plot_PDF_sample_mechanoadaptation_vs_passive_elliptic(
    #     nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360
    # )

    ### Dataset for ANN -- Mechano-adaptation vs passive elliptic ### 

    filename_qMC_Sobol_ANN_mechanoadaptation_vs_passive_elliptic = "dataset_for_ANN_mechanadaptation_vs_passive_elliptic.txt"

    samplerepresentativeness_ANN_mechanoadaptation_vs_passive_elliptic = SampleRepresentativeness(
        filename_qMC_Sobol_ANN_mechanoadaptation_vs_passive_elliptic, resampling_size=200, pixels=360)

    # samplerepresentativeness_ANN_mechanoadaptation_vs_passive_elliptic.compute_means_stds_of_shuffled_samples_and_export_to_pkl_ANN_mechanoadaptation_vs_passive_elliptic()

    samplerepresentativeness_ANN_mechanoadaptation_vs_passive_elliptic.plot_cumulative_mean_vs_sample_size_mechanoadaptation_vs_passive_elliptic(createfigure, savefigure, fonts)

    samplerepresentativeness_ANN_mechanoadaptation_vs_passive_elliptic.plot_cumulative_std_vs_sample_size_mechanoadaptation_vs_passive_elliptic(createfigure, savefigure, fonts)

    samplerepresentativeness_ANN_mechanoadaptation_vs_passive_elliptic.plot_gradient_cumulative_mean_vs_sample_size_mechanoadaptation_vs_passive_elliptic(createfigure, savefigure, fonts)

    samplerepresentativeness_ANN_mechanoadaptation_vs_passive_elliptic.plot_gradient_cumulative_std_vs_size_mechanoadaptation_vs_passive_elliptic(createfigure, savefigure, fonts)

    samplerepresentativeness_ANN_mechanoadaptation_vs_passive_elliptic.plot_PDF_sample_ANN_mechanoadaptation_vs_passive_elliptic(
        nb_bin, createfigure, savefigure, fonts, xticks, xticklabels, pixels=360
    )
