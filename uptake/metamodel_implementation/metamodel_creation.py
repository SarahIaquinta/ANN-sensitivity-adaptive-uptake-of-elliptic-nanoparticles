from pathlib import Path
import time
import numpy as np
import openturns as ot
ot.Log.Show(ot.Log.NONE)
from uptake.figures.utils import CreateFigure, Fonts, SaveFigure, XTickLabels, XTicks
import uptake.metamodel_implementation.utils as miu


class DataPreSetting:
    """A class that performs the presets on the dataset to compute the metamodel

    Attributes:
    ----------
    filename: string
        Name of the .txt file from which the data will be extracted
    training_amount : float
        Proportion (between 0 and 1) of the initial data used for training (the remaining data are
        used for testing)

    Methods:
    -------
    shuffle_dataset_from_datafile(self):
        Imports and shuffles the initial dataset
    extract_training_data_from_shuffled_dataset(self, shuffled_sample):
        Extracts a proportion (training_amount) of the shuffled_data to generate the training
        dataset
    extract_testing_data(self, shuffled_sample):
        Extracts the remaining data after the extraction of the training dataset for testing
    """

    def __init__(self, filename, training_amount):
        """Constructs all the necessary attributes for the DataPreSetting object.

        Parameters:
        ----------
        filename: string
            Name of the .txt file from which the data will be extracted
        training_amount : float
            Proportion (between 0 and 1) of the initial data used for training (the remaining
            data are used for testing)

        Returns:
        -------
        None
        """

        self.filename = Path.cwd() / "metamodel_implementation" / filename
        self.training_amount = training_amount

    def shuffle_dataset_from_datafile(self):
        """Imports and shuffles the initial dataset using Openturns library

        Parameters:
        ----------
        None

        Returns:
        -------
        sample: ot class
            Shuffled input dataset
        """

        sample = ot.Sample.ImportFromTextFile(self.filename.as_posix(), "\t", 0)
        np.random.shuffle(sample)
        return sample

    def extract_training_data_from_shuffled_dataset(self, shuffled_sample):
        """Extracts a proportion (training_amount) of the shuffled_data to train the metamodel

        Parameters:
        ----------
        shuffled_sample : ot class
            Shuffled input dataset

        Returns:
        -------
        shuffled_training_input_sample: ot class
            Part of the input data used to train the metamodel
        shuffled_training_output_sample: ot class
            Output data corresponding to the shuffled_training_input_sample
        """

        datasize, _ = np.shape(shuffled_sample)
        datasize = 1000
        shuffled_training_input_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), 0:3])
        shuffled_training_output_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), -1:])
        return shuffled_training_input_sample, shuffled_training_output_sample

    def extract_training_data_from_shuffled_dataset_constant_elliptic(self, shuffled_sample):
        """Extracts a proportion (training_amount) of the shuffled_data to train the metamodel

        Parameters:
        ----------
        shuffled_sample : ot class
            Shuffled input dataset

        Returns:
        -------
        shuffled_training_input_sample: ot class
            Part of the input data used to train the metamodel
        shuffled_training_output_sample: ot class
            Output data corresponding to the shuffled_training_input_sample
        """

        datasize, _ = np.shape(shuffled_sample)
        # datasize = 1000
        shuffled_training_input_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), 0:3])
        shuffled_training_output_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), -2])
        return shuffled_training_input_sample, shuffled_training_output_sample

    def extract_training_data_from_shuffled_dataset_mechanoadaptation_circular(self, shuffled_sample):
        """Extracts a proportion (training_amount) of the shuffled_data to train the metamodel

        Parameters:
        ----------
        shuffled_sample : ot class
            Shuffled input dataset

        Returns:
        -------
        shuffled_training_input_sample: ot class
            Part of the input data used to train the metamodel
        shuffled_training_output_sample: ot class
            Output data corresponding to the shuffled_training_input_sample
        """

        datasize, _ = np.shape(shuffled_sample)
        # datasize = 1000
        shuffled_training_input_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), 0:3])
        shuffled_training_output_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), -1])
        return shuffled_training_input_sample, shuffled_training_output_sample

    def extract_training_data_from_shuffled_dataset_mechanoadaptation_elliptic(self, shuffled_sample):
        """Extracts a proportion (training_amount) of the shuffled_data to train the metamodel

        Parameters:
        ----------
        shuffled_sample : ot class
            Shuffled input dataset

        Returns:
        -------
        shuffled_training_input_sample: ot class
            Part of the input data used to train the metamodel
        shuffled_training_output_sample: ot class
            Output data corresponding to the shuffled_training_input_sample
        """

        datasize, _ = np.shape(shuffled_sample)
        shuffled_training_input_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), 0:4])
        shuffled_training_output_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), -1])
        return shuffled_training_input_sample, shuffled_training_output_sample

    def extract_training_data_from_shuffled_dataset_mechanoadaptation_vs_passive_elliptic(self, shuffled_sample):
        """Extracts a proportion (training_amount) of the shuffled_data to train the metamodel

        Parameters:
        ----------
        shuffled_sample : ot class
            Shuffled input dataset

        Returns:
        -------
        shuffled_training_input_sample: ot class
            Part of the input data used to train the metamodel
        shuffled_training_output_sample: ot class
            Output data corresponding to the shuffled_training_input_sample
        """

        datasize, _ = np.shape(shuffled_sample)
        # datasize = 1000
        shuffled_training_input_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), 0:6])
        shuffled_training_output_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), -2])
        return shuffled_training_input_sample, shuffled_training_output_sample


    def extract_training_data_from_shuffled_dataset_mechanoadaptation_vs_passive_circular(self, shuffled_sample):
        """Extracts a proportion (training_amount) of the shuffled_data to train the metamodel

        Parameters:
        ----------
        shuffled_sample : ot class
            Shuffled input dataset

        Returns:
        -------
        shuffled_training_input_sample: ot class
            Part of the input data used to train the metamodel
        shuffled_training_output_sample: ot class
            Output data corresponding to the shuffled_training_input_sample
        """

        datasize, _ = np.shape(shuffled_sample)
        shuffled_training_input_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), 0:5])
        shuffled_training_output_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), -2])
        return shuffled_training_input_sample, shuffled_training_output_sample


    def extract_testing_data(self, shuffled_sample):
        """Extracts a proportion (1 - training_amount) of the shuffled_data to test the metamodel

        Parameters:
        ----------
        shuffled_sample : ot class
            Shuffled input dataset

        Returns:
        -------
        shuffled_testing_input_sample: ot class
            Part of the input data used to test the metamodel
        shuffled_testing_output_sample: ot class
            Output data corresponding to the shuffled_testing_input_sample
        """

        datasize, _ = np.shape(shuffled_sample)
        datasize = 1000
        shuffled_testing_input_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, 0:3]
        )
        shuffled_testing_output_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, -1:]
        )
        return shuffled_testing_input_sample, shuffled_testing_output_sample

    def extract_testing_data_constant_elliptic(self, shuffled_sample):
        """Extracts a proportion (1 - training_amount) of the shuffled_data to test the metamodel

        Parameters:
        ----------
        shuffled_sample : ot class
            Shuffled input dataset

        Returns:
        -------
        shuffled_testing_input_sample: ot class
            Part of the input data used to test the metamodel
        shuffled_testing_output_sample: ot class
            Output data corresponding to the shuffled_testing_input_sample
        """

        datasize, _ = np.shape(shuffled_sample)
        # datasize = 1000
        shuffled_testing_input_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, 0:3]
        )
        shuffled_testing_output_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, -2]
        )
        return shuffled_testing_input_sample, shuffled_testing_output_sample

    def extract_testing_data_mechanoadaptation_circular(self, shuffled_sample):
        """Extracts a proportion (1 - training_amount) of the shuffled_data to test the metamodel

        Parameters:
        ----------
        shuffled_sample : ot class
            Shuffled input dataset

        Returns:
        -------
        shuffled_testing_input_sample: ot class
            Part of the input data used to test the metamodel
        shuffled_testing_output_sample: ot class
            Output data corresponding to the shuffled_testing_input_sample
        """

        datasize, _ = np.shape(shuffled_sample)
        # datasize = 1000
        shuffled_testing_input_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, 0:3]
        )
        shuffled_testing_output_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, -1]
        )
        return shuffled_testing_input_sample, shuffled_testing_output_sample

    def extract_testing_data_mechanoadaptation_elliptic(self, shuffled_sample):
        """Extracts a proportion (1 - training_amount) of the shuffled_data to test the metamodel

        Parameters:
        ----------
        shuffled_sample : ot class
            Shuffled input dataset

        Returns:
        -------
        shuffled_testing_input_sample: ot class
            Part of the input data used to test the metamodel
        shuffled_testing_output_sample: ot class
            Output data corresponding to the shuffled_testing_input_sample
        """

        datasize, _ = np.shape(shuffled_sample)
        shuffled_testing_input_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, 0:4]
        )
        shuffled_testing_output_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, -1]
        )
        return shuffled_testing_input_sample, shuffled_testing_output_sample

    def extract_testing_data_mechanoadaptation_vs_passive_elliptic(self, shuffled_sample):
        """Extracts a proportion (1 - training_amount) of the shuffled_data to test the metamodel

        Parameters:
        ----------
        shuffled_sample : ot class
            Shuffled input dataset

        Returns:
        -------
        shuffled_testing_input_sample: ot class
            Part of the input data used to test the metamodel
        shuffled_testing_output_sample: ot class
            Output data corresponding to the shuffled_testing_input_sample
        """

        datasize, _ = np.shape(shuffled_sample)
        # datasize = 1000
        shuffled_testing_input_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, 0:6]
        )
        shuffled_testing_output_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, -2]
        )
        return shuffled_testing_input_sample, shuffled_testing_output_sample


    def extract_testing_data_mechanoadaptation_vs_passive_circular(self, shuffled_sample):
        """Extracts a proportion (1 - training_amount) of the shuffled_data to test the metamodel

        Parameters:
        ----------
        shuffled_sample : ot class
            Shuffled input dataset

        Returns:
        -------
        shuffled_testing_input_sample: ot class
            Part of the input data used to test the metamodel
        shuffled_testing_output_sample: ot class
            Output data corresponding to the shuffled_testing_input_sample
        """

        datasize, _ = np.shape(shuffled_sample)
        shuffled_testing_input_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, 0:5]
        )
        shuffled_testing_output_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, -2]
        )
        return shuffled_testing_input_sample, shuffled_testing_output_sample


class MetamodelCreation:
    """A class that builds the metamodel

    Attributes:
    ----------
    input_sample_training: ot class (array)
        Sample of input parameters for training, generated in the DataPreSetting class
    output_sample_training: ot class (array)
        Sample of output parameters for training, generated in the DataPreSetting class

    Methods:
    -------
    create_kriging_algorithm(self):
        Computes the Kriging (Gaussian Process) metamodel algorithm of the Openturns library
    create_pce_algorithm(self, degree):
        Computes the Polynomial Chaos Expansion (Functional Chaos) metamodel algorithm of the
        Openturns library
    """

    def __init__(self, input_sample_training, output_sample_training):
        """Constructs all the necessary attributes for the MetamodelCreation object.

        Parameters:
        ----------
        input_sample_training: ot class (array)
            sample of input parameters for training, generated in the DataPreSetting class
        output_sample_training: ot class (array)
            sample of output parameters for training, generated in the DataPreSetting class

        Returns:
        -------
        None
        """

        self.input_sample_training = input_sample_training
        self.output_sample_training = output_sample_training
        self.number_of_train_samples, self.dimension = np.shape(self.input_sample_training)

    def create_kriging_algorithm(self):
        """Computes the Kriging (Gaussian Process) metamodel algorithm of the Openturns library

        Parameters:
        ----------
        None

        Returns:
        -------
        kriging_algorithm: ot class
            Kriging algorithm from the Openturns library
        """

        basis = ot.Basis(0)
        cov = ot.SphericalModel(self.dimension)
        kriging_algorithm = ot.KrigingAlgorithm(self.input_sample_training, self.output_sample_training, cov, basis)
        return kriging_algorithm

    def compute_variogram(self):
        distances = np.zeros((int(self.number_of_train_samples * (self.number_of_train_samples -1))))
        y_distances = np.zeros_like(distances)
        for i in range(self.number_of_train_samples):
            remaining_samples_indices = [*range(0, i)] + [*range(i+1, self.number_of_train_samples)]
            for j in remaining_samples_indices:
                distance = miu.compute_distance(self.input_sample_training[i, :], self.input_sample_training[j, :])
                y_distance = np.abs(self.output_sample_training[i, :] - self.output_sample_training[j, :])
            distances[i] = distance
            y_distances[i] = y_distance
        return distances, y_distances

    def compute_and_export_variogram_constant_elliptic(self):
        distances, y_distances = self.compute_variogram()
        miu.export_variogram("variogram_constant_elliptic.pkl", distances, y_distances)

    def plot_variogram_constant_elliptic(self, fonts, createfigure, savefigure, pixels):
        distances, y_distances = miu.extract_variogram("variogram_constant_elliptic.pkl")
        fit_variogram_squared_exponential = miu.fit_variogram_squared_exponential(distances, 8, 0.4)
        fit_variogram_spherical = miu.fit_variogram_spherical(distances, 8, 0.4, 8)
        fig = createfigure.rectangle_figure(pixels=pixels)
        ax = fig.gca()
        ax.plot(distances, y_distances, 'x', 'b')
        ax.plot(distances, fit_variogram_squared_exponential, 'o', markersize=1, color='k', label = 'Squared Exponential')
        ax.plot(distances, fit_variogram_spherical, 'o', markersize=1, color='r', label = 'Spherical')
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_xticklabels([0, 2, 4, 6, 8], font=fonts.serif(), fontsize=fonts.axis_legend_size())
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], font=fonts.serif(), fontsize=fonts.axis_legend_size())
        ax.set_xlabel("$h$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"variation of $\tilde{f}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())

        savefigure.save_as_png(fig, "variogram_constant_elliptic" + str(pixels))

        

    def create_pce_algorithm(self, degree):
        """Computes the Polynomial Chaos Expansion (Functional Chaos) metamodel algorithm of the
            Openturns library

        Parameters:
        ----------
        None

        Returns:
        -------
        pce_algorithm: ot class
            PCE algorithm from the Openturns library
        """

        distribution = ot.ComposedDistribution([ot.Uniform(-1, 1), ot.Uniform(-1, 1), ot.Uniform(-1, 1)])

        gamma_bar_r_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 0])
        gamma_bar_fs_bar_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 1])
        gamma_bar_lambda_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 2])
        input_sample_training_rescaled = ot.Sample(len(gamma_bar_r_list_rescaled), 3)
        for k in range(len(gamma_bar_r_list_rescaled)):
            input_sample_training_rescaled[k, 0] = gamma_bar_r_list_rescaled[k]
            input_sample_training_rescaled[k, 1] = gamma_bar_fs_bar_list_rescaled[k]
            input_sample_training_rescaled[k, 2] = gamma_bar_lambda_list_rescaled[k]
        enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(3)
        productBasis = ot.OrthogonalBasis(
            ot.OrthogonalProductPolynomialFactory(
                [ot.LegendreFactory(), ot.LegendreFactory(), ot.LegendreFactory()], enumerateFunction
            )
        )
        indexMax = enumerateFunction.getStrataCumulatedCardinal(int(degree))
        adaptiveStrategy = ot.SequentialStrategy(productBasis, indexMax)
        sampling_size = 800
        experiment = ot.MonteCarloExperiment(sampling_size)
        projectionStrategy = ot.ProjectionStrategy(ot.LeastSquaresStrategy(experiment))
        pce_algorithm = ot.FunctionalChaosAlgorithm(
            input_sample_training_rescaled,
            self.output_sample_training,
            distribution,
            adaptiveStrategy,
            projectionStrategy,
        )
        return pce_algorithm

    def create_pce_algorithm_constant_elliptic(self, degree):
        """
        computes the pce Algorithm of the Openturns library

        Parameters:
            ----------
            None


        Returns:
            -------
            pce_algorithm: ot class
                pce algorithm from the Openturns library

        """

        gamma_bar_0_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 0])
        sigma_bar_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 1])
        r_bar_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 2])
        input_sample_training_rescaled  = ot.Sample(len(gamma_bar_0_list_rescaled), 3) 
        for k in range(len(gamma_bar_0_list_rescaled)):
            input_sample_training_rescaled[k, 0] = gamma_bar_0_list_rescaled[k]
            input_sample_training_rescaled[k, 1] = sigma_bar_list_rescaled[k]
            input_sample_training_rescaled[k, 2] = r_bar_list_rescaled[k]
        enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(3) 
        factory = ot.UserDefinedFactory()
        r_bar_distribution = factory.build(input_sample_training_rescaled[:, 2])   
        distribution = ot.ComposedDistribution(
            [
                ot.Uniform(-1, 1),
                ot.Uniform(-1, 1),
                r_bar_distribution            ]
        )
        # enumerateFunction = ot.LinearEnumerateFunction(3)        
        orthogonal_r_bar = ot.StandardDistributionPolynomialFactory(r_bar_distribution)

        productBasis  = ot.OrthogonalBasis(ot.OrthogonalProductPolynomialFactory([ot.LegendreFactory(), ot.LegendreFactory(), orthogonal_r_bar], enumerateFunction))
        indexMax = enumerateFunction.getStrataCumulatedCardinal(int(degree))
        # adaptiveStrategy = ot.CleaningStrategy(productBasis, indexMax)
        # adaptiveStrategy = ot.FixedStrategy(productBasis, indexMax)
        adaptiveStrategy = ot.SequentialStrategy(productBasis, indexMax)
        sampling_size = len(gamma_bar_0_list_rescaled)
        experiment = ot.MonteCarloExperiment(sampling_size)
        projectionStrategy = ot.ProjectionStrategy(ot.LeastSquaresStrategy(experiment))
        # projectionStrategy = ot.ProjectionStrategy(ot.IntegrationStrategy(distribution))
        pce_algorithm = ot.FunctionalChaosAlgorithm(input_sample_training_rescaled, self.output_sample_training, distribution, adaptiveStrategy, projectionStrategy)
        return pce_algorithm

    def create_pce_algorithm_mechanoadaptation_vs_passive_elliptic(self, degree):
        """
        computes the pce Algorithm of the Openturns library

        Parameters:
            ----------
            None


        Returns:
            -------
            pce_algorithm: ot class
                pce algorithm from the Openturns library

        """

        gamma_bar_0_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 0])
        sigma_bar_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 1])
        gamma_bar_r_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 2])
        gamma_bar_fs_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 3])
        gamma_bar_lambda_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 4])
        r_bar_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 5])
        input_sample_training_rescaled  = ot.Sample(len(gamma_bar_0_list_rescaled), 6) 
        for k in range(len(gamma_bar_0_list_rescaled)):
            input_sample_training_rescaled[k, 0] = gamma_bar_0_list_rescaled[k]
            input_sample_training_rescaled[k, 1] = sigma_bar_list_rescaled[k]
            input_sample_training_rescaled[k, 2] = gamma_bar_r_list_rescaled[k]
            input_sample_training_rescaled[k, 3] = gamma_bar_fs_list_rescaled[k]
            input_sample_training_rescaled[k, 4] = gamma_bar_lambda_list_rescaled[k]
            input_sample_training_rescaled[k, 5] = r_bar_list_rescaled[k]
        enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(6) 
        factory = ot.UserDefinedFactory()
        r_bar_distribution = factory.build(input_sample_training_rescaled[:, 2])   
        distribution = ot.ComposedDistribution(
            [
                ot.Uniform(-1, 1),
                ot.Uniform(-1, 1),
                ot.Uniform(-1, 1),
                ot.Uniform(-1, 1),
                ot.Uniform(-1, 1),
                r_bar_distribution            ]
        )
        # enumerateFunction = ot.LinearEnumerateFunction(3)        
        orthogonal_r_bar = ot.StandardDistributionPolynomialFactory(r_bar_distribution)

        productBasis  = ot.OrthogonalBasis(ot.OrthogonalProductPolynomialFactory([ot.LegendreFactory(), ot.LegendreFactory(), ot.LegendreFactory(), ot.LegendreFactory(), ot.LegendreFactory(), orthogonal_r_bar], enumerateFunction))
        indexMax = enumerateFunction.getStrataCumulatedCardinal(int(degree))
        # adaptiveStrategy = ot.CleaningStrategy(productBasis, indexMax)
        # adaptiveStrategy = ot.FixedStrategy(productBasis, indexMax)
        adaptiveStrategy = ot.SequentialStrategy(productBasis, indexMax)
        sampling_size = len(gamma_bar_0_list_rescaled)
        experiment = ot.MonteCarloExperiment(sampling_size)
        projectionStrategy = ot.ProjectionStrategy(ot.LeastSquaresStrategy(experiment))
        # projectionStrategy = ot.ProjectionStrategy(ot.IntegrationStrategy(distribution))
        pce_algorithm = ot.FunctionalChaosAlgorithm(input_sample_training_rescaled, self.output_sample_training, distribution, adaptiveStrategy, projectionStrategy)
        return pce_algorithm

    def create_pce_algorithm_mechanoadaptation_vs_passive_circular(self, degree):
        """Computes the Polynomial Chaos Expansion (Functional Chaos) metamodel algorithm of the
            Openturns library

        Parameters:
        ----------
        None

        Returns:
        -------
        pce_algorithm: ot class
            PCE algorithm from the Openturns library
        """

        distribution = ot.ComposedDistribution([ot.Uniform(-1, 1), ot.Uniform(-1, 1), ot.Uniform(-1, 1), ot.Uniform(-1, 1), ot.Uniform(-1, 1)])

        gamma_bar_0_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 0])
        sigma_bar_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 1])
        gamma_bar_r_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 2])
        gamma_bar_fs_bar_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 3])
        gamma_bar_lambda_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 4])
        input_sample_training_rescaled = ot.Sample(len(gamma_bar_r_list_rescaled), 5)
        for k in range(len(gamma_bar_r_list_rescaled)):
            input_sample_training_rescaled[k, 0] = gamma_bar_0_list_rescaled[k]
            input_sample_training_rescaled[k, 1] = sigma_bar_list_rescaled[k]
            input_sample_training_rescaled[k, 2] = gamma_bar_r_list_rescaled[k]
            input_sample_training_rescaled[k, 3] = gamma_bar_fs_bar_list_rescaled[k]
            input_sample_training_rescaled[k, 4] = gamma_bar_lambda_list_rescaled[k]
        enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(5)
        productBasis = ot.OrthogonalBasis(
            ot.OrthogonalProductPolynomialFactory(
                [ot.LegendreFactory(), ot.LegendreFactory(), ot.LegendreFactory(), ot.LegendreFactory(), ot.LegendreFactory()], enumerateFunction
            )
        )
        indexMax = enumerateFunction.getStrataCumulatedCardinal(int(degree))
        adaptiveStrategy = ot.SequentialStrategy(productBasis, indexMax)
        sampling_size = 800
        experiment = ot.MonteCarloExperiment(sampling_size)
        projectionStrategy = ot.ProjectionStrategy(ot.LeastSquaresStrategy(experiment))
        pce_algorithm = ot.FunctionalChaosAlgorithm(
            input_sample_training_rescaled,
            self.output_sample_training,
            distribution,
            adaptiveStrategy,
            projectionStrategy,
        )
        return pce_algorithm

    def create_pce_algorithm_mechanoadaptation_circular(self, degree):
        """Computes the Polynomial Chaos Expansion (Functional Chaos) metamodel algorithm of the
            Openturns library

        Parameters:
        ----------
        None

        Returns:
        -------
        pce_algorithm: ot class
            PCE algorithm from the Openturns library
        """

        distribution = ot.ComposedDistribution([ ot.Uniform(-1, 1), ot.Uniform(-1, 1), ot.Uniform(-1, 1)])

        gamma_bar_r_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 0])
        gamma_bar_fs_bar_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 1])
        gamma_bar_lambda_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 2])
        input_sample_training_rescaled = ot.Sample(len(gamma_bar_r_list_rescaled), 3)
        for k in range(len(gamma_bar_r_list_rescaled)):
            input_sample_training_rescaled[k, 0] = gamma_bar_r_list_rescaled[k]
            input_sample_training_rescaled[k, 1] = gamma_bar_fs_bar_list_rescaled[k]
            input_sample_training_rescaled[k, 2] = gamma_bar_lambda_list_rescaled[k]

        enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(3)
        productBasis = ot.OrthogonalBasis(
            ot.OrthogonalProductPolynomialFactory(
                [ot.LegendreFactory(), ot.LegendreFactory(), ot.LegendreFactory()], enumerateFunction
            )
        )
        indexMax = enumerateFunction.getStrataCumulatedCardinal(int(degree))
        adaptiveStrategy = ot.SequentialStrategy(productBasis, indexMax)
        sampling_size = 800#len(gamma_bar_r_list_rescaled)
        experiment = ot.MonteCarloExperiment(sampling_size)
        projectionStrategy = ot.ProjectionStrategy(ot.LeastSquaresStrategy(experiment))
        pce_algorithm = ot.FunctionalChaosAlgorithm(
            input_sample_training_rescaled,
            self.output_sample_training,
            distribution,
            adaptiveStrategy,
            projectionStrategy,
        )
        return pce_algorithm

    def create_pce_algorithm_mechanoadaptation_elliptic(self, degree):
        """Computes the Polynomial Chaos Expansion (Functional Chaos) metamodel algorithm of the
            Openturns library

        Parameters:
        ----------
        None

        Returns:
        -------
        pce_algorithm: ot class
            PCE algorithm from the Openturns library
        """

        gamma_bar_r_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 0])
        gamma_bar_fs_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 1])
        gamma_bar_lambda_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 2])
        r_bar_list_rescaled = miu.rescale_sample(self.input_sample_training[:, 3])
        input_sample_training_rescaled  = ot.Sample(len(gamma_bar_r_list_rescaled), 4) 
        for k in range(len(gamma_bar_r_list_rescaled)):
            input_sample_training_rescaled[k, 0] = gamma_bar_r_list_rescaled[k]
            input_sample_training_rescaled[k, 1] = gamma_bar_fs_list_rescaled[k]
            input_sample_training_rescaled[k, 2] = gamma_bar_lambda_list_rescaled[k]
            input_sample_training_rescaled[k, 3] = r_bar_list_rescaled[k]
        enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(4) 
        factory = ot.UserDefinedFactory()
        r_bar_distribution = factory.build(input_sample_training_rescaled[:, 3])   
        distribution = ot.ComposedDistribution(
            [
                ot.Uniform(-1, 1),
                ot.Uniform(-1, 1),
                ot.Uniform(-1, 1),
                r_bar_distribution            ]
        )
        # enumerateFunction = ot.LinearEnumerateFunction(3)        
        orthogonal_r_bar = ot.StandardDistributionPolynomialFactory(r_bar_distribution)

        productBasis  = ot.OrthogonalBasis(ot.OrthogonalProductPolynomialFactory([ot.LegendreFactory(), ot.LegendreFactory(), ot.LegendreFactory(), orthogonal_r_bar], enumerateFunction))
        indexMax = enumerateFunction.getStrataCumulatedCardinal(int(degree))
        # adaptiveStrategy = ot.CleaningStrategy(productBasis, indexMax)
        # adaptiveStrategy = ot.FixedStrategy(productBasis, indexMax)
        adaptiveStrategy = ot.SequentialStrategy(productBasis, indexMax)
        sampling_size = len(gamma_bar_r_list_rescaled)
        experiment = ot.MonteCarloExperiment(sampling_size)
        projectionStrategy = ot.ProjectionStrategy(ot.LeastSquaresStrategy(experiment))
        pce_algorithm = ot.FunctionalChaosAlgorithm(
            input_sample_training_rescaled,
            self.output_sample_training,
            distribution,
            adaptiveStrategy,
            projectionStrategy,
        )
        return pce_algorithm


class MetamodelPostTreatment:
    """A class that extracts and exports the features of the metamodel

    Attributes:
    ----------
    None

    Methods:
    -------
    run_algorithm(self, algorithm):
        Runs the algorithm
    extract_results_from_algorithm(self, algorithm):
        Gets the results from the algorithm
    get_metamodel_from_results_algo(self, results_from_algo):
        Gets the metamodel from the algorithm's results
    get_errors_from_metamodel(self, results_from_algo):
        Gets the errors from the metamodel (obtained according to the results_from_algo)
    """

    def __init__(self):
        """Constructs all the necessary attributes for the MetamodelPostTreatment object.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

    def run_algorithm(self, algorithm):
        """Runs the algorithm

        Parameters:
        ----------
        algorithm: ot class
            Algorithm ("kriging" or "pce"), output of the methods which creates the respective
            algorithm

        Returns:
        -------
        None
        """

        algorithm.run()

    def extract_results_from_algorithm(self, algorithm):
        """Gets the results from the algorithm

        Parameters:
        ----------
        algorithm: ot class
            Algorithm (kriging or pce), output of the methods which creates the respective algorithm

        Returns:
        -------
        results: ot class
            Result from the algorithm, computing following a method from the Openturns library

        """
        results = algorithm.getResult()
        return results

    def get_metamodel_from_results_algo(self, results_from_algo):
        """Gets the metamodel from the algorithm's results

        Parameters:
        ----------
        results_from_algo: ot class
            Result from the algorithm, computing following a method from the Openturns library
            output of the method self.extract_results_from_algorithm(algorithm)

        Returns:
        -------
        metamodel: ot class
            Metamodel computed by the algorithm. contains the information necessary to use the
            metamodel
        """

        metamodel = results_from_algo.getMetaModel()
        return metamodel

    def get_errors_from_metamodel(self, results_from_algo):
        """Gets the errors from the metamodel (obtained according to the results_from_algo)

        Parameters:
        ----------
        results_from_algo: ot class
            Result from the algorithm, computing following a method from the Openturns library

        Returns:
        -------
        residual: ot class (array)
            Residual of the metamodel
        relative_error: ot class (array)
            Relative_error of the metamodel
        """

        residual = results_from_algo.getResiduals()
        relative_error = results_from_algo.getRelativeErrors()
        return residual, relative_error


def metamodel_creation_routine_kriging(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel

    Returns:
    -------
    None
    """

    kri = metamodelcreation.create_kriging_algorithm()
    metamodelposttreatment.run_algorithm(kri)
    results_from_kri = metamodelposttreatment.extract_results_from_algorithm(kri)
    complete_pkl_filename_Kriging = miu.create_pkl_name("Kriging_mechanoadaptation_vs_passive_circular", datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_kri, complete_pkl_filename_Kriging)

def metamodel_creation_routine_kriging_constant_elliptic(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel

    Returns:
    -------
    None
    """

    kri = metamodelcreation.create_kriging_algorithm()
    metamodelposttreatment.run_algorithm(kri)
    results_from_kri = metamodelposttreatment.extract_results_from_algorithm(kri)
    complete_pkl_filename_Kriging = miu.create_pkl_name("Kriging_constant_elliptic", datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_kri, complete_pkl_filename_Kriging)

def metamodel_creation_routine_kriging_mechanoadaptation_vs_passive_elliptic(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel

    Returns:
    -------
    None
    """

    kri = metamodelcreation.create_kriging_algorithm()
    metamodelposttreatment.run_algorithm(kri)
    results_from_kri = metamodelposttreatment.extract_results_from_algorithm(kri)
    complete_pkl_filename_Kriging = miu.create_pkl_name("Kriging_mechanoadaptation_vs_passive_elliptic", datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_kri, complete_pkl_filename_Kriging)

def metamodel_creation_routine_kriging_mechanoadaptation_vs_passive_elliptic_new_settings(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel

    Returns:
    -------
    None
    """

    kri = metamodelcreation.create_kriging_algorithm()
    metamodelposttreatment.run_algorithm(kri)
    results_from_kri = metamodelposttreatment.extract_results_from_algorithm(kri)
    complete_pkl_filename_Kriging = miu.create_pkl_name("Kriging_mechanoadaptation_vs_passive_elliptic_new_settings_article", datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_kri, complete_pkl_filename_Kriging)


def metamodel_creation_routine_kriging_mechanoadaptation_vs_passive_circular(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel

    Returns:
    -------
    None
    """

    kri = metamodelcreation.create_kriging_algorithm()
    metamodelposttreatment.run_algorithm(kri)
    results_from_kri = metamodelposttreatment.extract_results_from_algorithm(kri)
    complete_pkl_filename_Kriging = miu.create_pkl_name("Kriging_mechanoadaptation_vs_passive_circular", datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_kri, complete_pkl_filename_Kriging)

def metamodel_creation_routine_kriging_mechanoadaptation_circular(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel

    Returns:
    -------
    None
    """

    kri = metamodelcreation.create_kriging_algorithm()
    metamodelposttreatment.run_algorithm(kri)
    results_from_kri = metamodelposttreatment.extract_results_from_algorithm(kri)
    complete_pkl_filename_Kriging = miu.create_pkl_name("Kriging_mechanoadaptation_circular", datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_kri, complete_pkl_filename_Kriging)

def metamodel_creation_routine_kriging_mechanoadaptation_elliptic(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel

    Returns:
    -------
    None
    """

    kri = metamodelcreation.create_kriging_algorithm()
    metamodelposttreatment.run_algorithm(kri)
    results_from_kri = metamodelposttreatment.extract_results_from_algorithm(kri)
    complete_pkl_filename_Kriging = miu.create_pkl_name("Kriging_mechanoadaptation_elliptic", datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_kri, complete_pkl_filename_Kriging)


def metamodel_creation_routine_pce(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel
    degree: int
        Dimension of the basis of polynoms

    Returns:
    -------
    Nothing

    """
    pce = metamodelcreation.create_pce_algorithm(degree)
    metamodelposttreatment.run_algorithm(pce)
    results_from_pce = metamodelposttreatment.extract_results_from_algorithm(pce)
    complete_pkl_filename_pce = miu.create_pkl_name("PCE" + str(degree), datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_pce, complete_pkl_filename_pce)

def metamodel_creation_routine_pce_constant_elliptic(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel
    degree: int
        Dimension of the basis of polynoms

    Returns:
    -------
    Nothing

    """
    pce = metamodelcreation.create_pce_algorithm_constant_elliptic(degree)
    metamodelposttreatment.run_algorithm(pce)
    results_from_pce = metamodelposttreatment.extract_results_from_algorithm(pce)
    complete_pkl_filename_pce = miu.create_pkl_name("PCE_constant_elliptic" + str(degree), datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_pce, complete_pkl_filename_pce)

def metamodel_creation_routine_pce_mechanoadaptation_vs_passive_elliptic(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel
    degree: int
        Dimension of the basis of polynoms

    Returns:
    -------
    Nothing

    """
    pce = metamodelcreation.create_pce_algorithm_mechanoadaptation_vs_passive_elliptic(degree)
    metamodelposttreatment.run_algorithm(pce)
    results_from_pce = metamodelposttreatment.extract_results_from_algorithm(pce)
    complete_pkl_filename_pce = miu.create_pkl_name("PCE_mechanoadaptation_vs_passive_elliptic" + str(degree), datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_pce, complete_pkl_filename_pce)

def metamodel_creation_routine_pce_mechanoadaptation_vs_passive_elliptic_new_settings(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel
    degree: int
        Dimension of the basis of polynoms

    Returns:
    -------
    Nothing

    """
    pce = metamodelcreation.create_pce_algorithm_mechanoadaptation_vs_passive_elliptic(degree)
    metamodelposttreatment.run_algorithm(pce)
    results_from_pce = metamodelposttreatment.extract_results_from_algorithm(pce)
    complete_pkl_filename_pce = miu.create_pkl_name("PCE_mechanoadaptation_vs_passive_elliptic_new_settings_article" + str(degree), datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_pce, complete_pkl_filename_pce)


def metamodel_creation_routine_pce_mechanoadaptation_circular(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel
    degree: int
        Dimension of the basis of polynoms

    Returns:
    -------
    Nothing

    """
    pce = metamodelcreation.create_pce_algorithm_mechanoadaptation_circular(degree)
    metamodelposttreatment.run_algorithm(pce)
    results_from_pce = metamodelposttreatment.extract_results_from_algorithm(pce)
    complete_pkl_filename_pce = miu.create_pkl_name("PCE_mechanoadaptation_circular" + str(degree), datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_pce, complete_pkl_filename_pce)

def metamodel_creation_routine_pce_mechanoadaptation_circular_new_settings(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel
    degree: int
        Dimension of the basis of polynoms

    Returns:
    -------
    Nothing

    """
    pce = metamodelcreation.create_pce_algorithm_mechanoadaptation_circular(degree)
    metamodelposttreatment.run_algorithm(pce)
    results_from_pce = metamodelposttreatment.extract_results_from_algorithm(pce)
    complete_pkl_filename_pce = miu.create_pkl_name("PCE_mechanoadaptation_circular_new_settings" + str(degree), datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_pce, complete_pkl_filename_pce)


def metamodel_creation_routine_pce_mechanoadaptation_elliptic(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel
    degree: int
        Dimension of the basis of polynoms

    Returns:
    -------
    Nothing

    """
    pce = metamodelcreation.create_pce_algorithm_mechanoadaptation_elliptic(degree)
    metamodelposttreatment.run_algorithm(pce)
    results_from_pce = metamodelposttreatment.extract_results_from_algorithm(pce)
    complete_pkl_filename_pce = miu.create_pkl_name("PCE_mechanoadaptation_elliptic" + str(degree), datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_pce, complete_pkl_filename_pce)


def metamodel_creation_routine_pce_mechanoadaptation_vs_passive_circular(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree):
    """Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel
        2 - creates the metamodel
        3 - exports the metamodel in a .pkl file

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelcreation: class
        A class that contains the routines to create a metamodel
    metamodelposttreatment: class
        A class that contains the routines to extract the algorithms from the metamodel
    shuffled_sample: ot class
        Sample that was used to create the metamodel
    degree: int
        Dimension of the basis of polynoms

    Returns:
    -------
    Nothing

    """
    pce = metamodelcreation.create_pce_algorithm_mechanoadaptation_vs_passive_circular(degree)
    metamodelposttreatment.run_algorithm(pce)
    results_from_pce = metamodelposttreatment.extract_results_from_algorithm(pce)
    complete_pkl_filename_pce = miu.create_pkl_name("PCE_mechanoadaptation_vs_passive_circular" + str(degree), datapresetting.training_amount)
    miu.export_metamodel_and_data_to_pkl(shuffled_sample, results_from_pce, complete_pkl_filename_pce)


if __name__ == "__main__":
    metamodelposttreatment = MetamodelPostTreatment()
    fonts = Fonts()
    createfigure = CreateFigure()
    savefigure = SaveFigure()
    pixels = 360


    ### Constant elliptic ###
    
    # filename_qMC_constant_elliptic = "dataset_for_metamodel_creation_feq_constant_elliptic.txt"
    # training_amount_list_constant_elliptic = [0.9]
    # degree_list_constant_elliptic = np.arange(1, 21)

    
    # for training_amount in training_amount_list_constant_elliptic:
    #     datapresetting = DataPreSetting(filename_qMC_constant_elliptic, training_amount)
    #     shuffled_sample = datapresetting.shuffle_dataset_from_datafile()
    #     input_sample_training, output_sample_training = datapresetting.extract_training_data_from_shuffled_dataset_constant_elliptic(
    #         shuffled_sample
    #     )
    #     metamodelcreation = MetamodelCreation(input_sample_training, output_sample_training)
        # metamodelcreation.compute_and_export_variogram_constant_elliptic()
        # metamodelcreation.plot_variogram_constant_elliptic(fonts, createfigure, savefigure, pixels)
        # metamodel_creation_routine_kriging_constant_elliptic(
        #     datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample
        # )

    #     for degree in degree_list_constant_elliptic:
    #         metamodel_creation_routine_pce_constant_elliptic(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree)

    ### Mechano-adaptation circular ###
    
    # filename_qMC_mechanoadaptation_circular = "dataset_for_metamodel_creation_mechanoadaptation_circular.txt"
    # training_amount_list_mechanoadaptation_circular = [0.8]
    # degree_list_mechanoadaptation_circular = np.arange(1, 12)

    
    # for training_amount in training_amount_list_mechanoadaptation_circular:
    #     datapresetting = DataPreSetting(filename_qMC_mechanoadaptation_circular, training_amount)
    #     shuffled_sample = datapresetting.shuffle_dataset_from_datafile()
    #     input_sample_training, output_sample_training = datapresetting.extract_training_data_from_shuffled_dataset_mechanoadaptation_circular(
    #         shuffled_sample
    #     )
    #     metamodelcreation = MetamodelCreation(input_sample_training, output_sample_training)
    #     metamodel_creation_routine_kriging_mechanoadaptation_circular(
    #         datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample
    #     )

    # #     for degree in degree_list_mechanoadaptation_circular:
    # #         metamodel_creation_routine_pce_mechanoadaptation_circular(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree)

    ### Mechano-adaptation vs passive circular ###
    
    # filename_qMC_mechanoadaptation_vs_passive_circular = "dataset_for_metamodel_creation_mechanoadaptation_vs_passive_circular.txt"
    # training_amount_list_mechanoadaptation_vs_passive_circular = [0.8]
    # degree_list_mechanoadaptation_vs_passive_circular = np.arange(1, 8)

    
    # for training_amount in training_amount_list_mechanoadaptation_vs_passive_circular:
    #     datapresetting = DataPreSetting(filename_qMC_mechanoadaptation_vs_passive_circular, training_amount)
    #     shuffled_sample = datapresetting.shuffle_dataset_from_datafile()
    #     input_sample_training, output_sample_training = datapresetting.extract_training_data_from_shuffled_dataset_mechanoadaptation_vs_passive_circular(
    #         shuffled_sample
    #     )
    #     metamodelcreation = MetamodelCreation(input_sample_training, output_sample_training)
    #     metamodel_creation_routine_kriging_mechanoadaptation_vs_passive_circular(
    #         datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample
    #     )

    # #     for degree in degree_list_mechanoadaptation_vs_passive_circular:
    # #         metamodel_creation_routine_pce_mechanoadaptation_vs_passive_circular(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree)

    ## Mechano-adaptation elliptic ###
    
    # filename_qMC_mechanoadaptation_elliptic = "dataset_for_metamodel_creation_mechanoadaptation_elliptic.txt"
    # training_amount_list_mechanoadaptation_elliptic = [0.65]
    # degree_list_mechanoadaptation_elliptic = np.arange(1, 7)

    
    # for training_amount in training_amount_list_mechanoadaptation_elliptic:
    #     datapresetting = DataPreSetting(filename_qMC_mechanoadaptation_elliptic, training_amount)
    #     shuffled_sample = datapresetting.shuffle_dataset_from_datafile()
    #     input_sample_training, output_sample_training = datapresetting.extract_training_data_from_shuffled_dataset_mechanoadaptation_elliptic(
    #         shuffled_sample
    #     )
    #     metamodelcreation = MetamodelCreation(input_sample_training, output_sample_training)
    #     metamodel_creation_routine_kriging_mechanoadaptation_elliptic(
    #         datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample
    #     )

    #     # for degree in degree_list_mechanoadaptation_elliptic:
    #     #     metamodel_creation_routine_pce_mechanoadaptation_elliptic(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree)

    ## Mechano-adaptation vs passive elliptic ###
    
    filename_qMC_mechanoadaptation_vs_passive_elliptic = "dataset_for_metamodel_creation_mechanoadaptation_vs_passive_elliptic.txt"
    training_amount_list_mechanoadaptation_vs_passive_elliptic = [0.9]
    degree_list_mechanoadaptation_vs_passive_elliptic = np.arange(1, 6)

    
    for training_amount in training_amount_list_mechanoadaptation_vs_passive_elliptic:
        datapresetting = DataPreSetting(filename_qMC_mechanoadaptation_vs_passive_elliptic, training_amount)
        shuffled_sample = datapresetting.shuffle_dataset_from_datafile()
        input_sample_training, output_sample_training = datapresetting.extract_training_data_from_shuffled_dataset_mechanoadaptation_vs_passive_elliptic(
            shuffled_sample
        )
        metamodelcreation = MetamodelCreation(input_sample_training, output_sample_training)
        metamodel_creation_routine_kriging_mechanoadaptation_vs_passive_elliptic(
            datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample
        )

    #     # for degree in degree_list_mechanoadaptation_vs_passive_elliptic:
    #     #     metamodel_creation_routine_pce_mechanoadaptation_vs_passive_elliptic(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample, degree)

