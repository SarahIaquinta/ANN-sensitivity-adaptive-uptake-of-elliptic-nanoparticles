import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from uptake import make_data_folder


class Files:
    """
    A class define the .pkl files to import.

    Attributes:
        ----------
        beginning: string
            beginning of the pkl file to be extracted

    Methods:
        -------
        import_files(self):
            Returns a list of the imported the .pkl files
        extraction_ellipsoid(self, outfile):
            Returns the useful data from each .pkl file

    """

    def __init__(self, beginning):
        """
        Constructs all the necessary attributes for the mechanical properties object.

        Parameters:
            ----------
            beginning: string
                beginning of the pkl file to be extracted

        Returns:
            -------
            None
        """

        self.beginning = beginning

    def import_files(self, folder):
        """
        Imports the .pkl files

        Parameters:
            ----------
            beginning: string
                beginning of the pkl file to be extracted

        Returns:
            -------
            outfile_list: list
                list of .pkl files starting by "beginning"

        """
        path_to_data_folder = make_data_folder(folder)
        entries = os.listdir(path_to_data_folder)
        len_beginning = len(self.beginning)
        outfile_list = []
        for i in range(len(entries)):
            outfile = entries[i]
            if outfile[0:len_beginning] == self.beginning:
                outfile_list.append(outfile)
        return outfile_list

    def extract_data_from_file(self, outfile, folder):
        """
        Extracts the useful data from each .pkl file

        Parameters:
            ----------
            outfile: string
                name of the .pkl file extracted from import_files function

        Returns:
            -------
            particle: class
                Particle class, as defined in system_definition.py file
            mechanics: class
                Mechanics class, as defined in system_definition.py file
            membrane: class
                Membrane class, as defined in system_definition.py file
            wrapping: class
                Wrapping class, as defined in system_definition.py file
            energy_list: list
                list of the adimensional energy variation for every wrapping degree
            time_list: list
                list of process time to compute each energy variaiton step
            f_eq: float
                wrapping at equilibrium
            wrapping_phase: string
                name of the wrapping phase at equilibrium
                    (no wrapping, partial wrapping or full wrapping)
            wrapping_phase_number: float
                number of the wrapping phase at equilibrium (1, 2 or 3)

        """
        path_to_data_folder = make_data_folder(folder)
        complete_filename = path_to_data_folder / outfile
        with open(complete_filename, "rb") as f:
            [
                particle,
                mechanics,
                membrane,
                wrapping,
                energy_list,
                time_list,
                f_eq,
                wrapping_phase,
                wrapping_phase_number,
            ] = pickle.load(f)
        return (
            particle,
            mechanics,
            membrane,
            wrapping,
            energy_list,
            time_list,
            f_eq,
            wrapping_phase,
            wrapping_phase_number,
        )

    def extract_data_for_phase_diagram(self, folder):
        outfile_list = self.import_files(folder)
        amount_of_files = len(outfile_list)
        phase_storage = np.zeros((amount_of_files, 3))
        for i in range(amount_of_files):
            outfile = outfile_list[i]
            _, mechanics, _, _, _, _, _, _, wrapping_phase_number = self.extract_data_from_file(outfile, folder)
            phase_storage[i, 0] = mechanics.gamma_bar_0
            phase_storage[i, 1] = mechanics.sigma_bar
            phase_storage[i, 2] = int(wrapping_phase_number)
        return phase_storage

    def extract_phase_proportions(self, folder):
        phase_storage = self.extract_data_for_phase_diagram(folder)
        phase_list = phase_storage[:, 2]
        phase1 = np.where(phase_list == 1)[0]
        phase2 = np.where(phase_list == 2)[0]
        phase3 = np.where(phase_list == 3)[0]
        total_phase1 = len(phase1)
        total_phase2 = len(phase2)
        total_phase3 = len(phase3)
        total_points = total_phase1 + total_phase2 + total_phase3
        proportion_phase1 = np.round(total_phase1 / total_points, 3)
        proportion_phase2 = np.round(total_phase2 / total_points, 3)
        proportion_phase3 = np.round(total_phase3 / total_points, 3)
        return proportion_phase1, proportion_phase2, proportion_phase3


class Plot:
    """
    A class to gather the plot functions

    Attributes:
        ----------
        None

    Methods:
        -------
        adimensional_energy_variation(self, particle, mechanics, wrapping, energy_list):
            Plots the variation of adimensional energy with respect to wrapping
        phase_diagram(self, files):
            Plots the phase diagram for the files extracted in the files class

    """

    def adimensional_energy_variation(self, particle, mechanics, wrapping, energy_list):
        plt.plot(
            wrapping.wrapping_list,
            energy_list,
            "-",
            label=r"$\overline{r} = $ "
            + str(particle.r_bar)
            + "\n$\\overline{\\gamma}_r$ = "
            + str(mechanics.gamma_bar_r)
            + r" ; $\overline{\gamma}_{fs}$ = "
            + str(mechanics.gamma_bar_fs)
            + r" ; $\overline{\gamma}_{\lambda}$ = "
            + str(mechanics.gamma_bar_lambda)
            + r" ; $\overline{\gamma}_0$ = "
            + str(mechanics.gamma_bar_0)
            + "\n$\\overline{\\sigma}_r$ = "
            + str(mechanics.sigma_bar_r)
            + r" ; $\overline{\sigma}_{fs}$ = "
            + str(mechanics.sigma_bar_fs)
            + r" ; $\overline{\sigma}_{\lambda}$ = "
            + str(mechanics.sigma_bar_lambda)
            + r" ; $\overline{\sigma}_0$ = "
            + str(mechanics.sigma_bar_0),
        )
        plt.legend()
        plt.xlabel("wrapping degree f [-]")
        plt.ylabel(r"$\Delta E [-]$")
        plt.title(r"$\Delta E(f)$")
        plt.xlim((0, 1))


if __name__ == "__main__":
    print("hello")
