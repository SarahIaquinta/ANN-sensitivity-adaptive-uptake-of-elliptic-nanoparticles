import multiprocessing as mp
import os
import pickle
import time
from datetime import timedelta
from functools import partial
from os import listdir
from os.path import isfile, join
from pathlib import Path

import numpy as np

import uptake.model.utils
import uptake.multiprocessing.datasets_definition as datadef
import uptake.multiprocessing.utils as mputils
from uptake import make_data_folder
from uptake.model.cellular_uptake_rigid_particle import EnergyComputation, identify_wrapping_phase
from uptake.model.system_definition import MembraneGeometry
from uptake.model_posttreatment.separate_data_from_sensitivity_analysis import (
    generate_and_export_feq,
    generate_and_export_feq_allvar,
    generate_and_export_phase_proportions,
)


def compute_energy_and_get_phase_for_given_mechanics_and_particle(folder_name, zip_i):
    """
    Args:
        mechanics_i : class - mechanics class from MechanicalProperties_Adaptation
    Returns:
        Nothing
        Creates a .pkl file with the outputs of the simulations, as follows:
            particle : class - from ParticleGeometry
            mechanics_i : class - the input parameter of the function
            membrane : class - from MembraneGeometry
            wrapping : class - from Wrapping
            energy_list : list - evolution of the total energy variation with wrapping
            time_list : list - CPU time needed to compute energy at each step of wrapping
            f_eq : float - wrapping degree at equilibrium
            wrapping_phase : string - name of wrapping phase at equilibrium
            wrapping_phase_number : int - number id corresponding to thw wrapping_phase
    """
    path_to_data_folder = make_data_folder(folder_name)
    # print(path_to_data_folder)
    existing_filenames = {f for f in listdir(path_to_data_folder) if isfile(join(path_to_data_folder, f))}
    wrapping = mputils.define_global_Wrapping_class()
    energy_computation = EnergyComputation()
    mechanics_i, particle_i = list(zip_i)[0], list(zip_i)[1]
    membrane_i = MembraneGeometry(particle_i, sampling_points_membrane=300)
    outfile_name = uptake.model.utils.define_pkl_filename(particle_i, mechanics_i)
    # print(outfile_name)
    if outfile_name not in existing_filenames:
        (f_eq, wrapping_phase_number, wrapping_phase, energy_list, time_list, _, _, _, _) = identify_wrapping_phase(
            particle_i, mechanics_i, membrane_i, wrapping, energy_computation
        )
        complete_filename = path_to_data_folder / outfile_name
        with open(complete_filename, "wb") as f:
            pickle.dump(
                [
                    particle_i,
                    mechanics_i,
                    membrane_i,
                    wrapping,
                    energy_list,
                    time_list,
                    f_eq,
                    wrapping_phase,
                    wrapping_phase_number,
                ],
                f,
            )


def parallelize_energy_computation_and_phase_determination_with_mechanics_and_particle(func, zip_generator, CPUs):
    """
    Args:
        zip_generator : generator - generator of the mechanics classes set to test
    Returns:
        Nothing
        runs the calculations in parallel with max_workers computation hearts
    """
    with mp.Pool(CPUs) as executor:  # the argument of the mp.Pool() function is the amount of CPU to parallelize
        executor.map(func, zip_generator)


def run_parallel_routine_to_get_phases_given_mechanics_and_particle(testcase_id, zip_generator, CPUs):
    """
    Args:
        testcase_id : string - name of the investigated testcase
        mechanics_generator : generator - generator of the mechanics classes set to test
        CPUs : int - amount of CPUs used to paralellize the tasks
    Returns:
        Nothing
        runs the calculations in parallel with CPUs computation cores and
        generates the file containing the phase proportions from the phase diagrams
    """
    print(testcase_id, "STARTED")
    print(f"computing on {CPUs} CPUs")
    start = time.perf_counter()
    func = partial(compute_energy_and_get_phase_for_given_mechanics_and_particle, testcase_id)
    parallelize_energy_computation_and_phase_determination_with_mechanics_and_particle(func, zip_generator, CPUs)
    print(testcase_id, " : GENERATED ALL PKL FILES")
    generate_and_export_phase_proportions(testcase_id, folder=testcase_id)
    print(testcase_id, "GENERATED DATAFILE")
    end = time.perf_counter()
    elapsed = end - start
    print(testcase_id, "DONE")
    elapsed_time = timedelta(seconds=elapsed)
    print(f"elapsed time: {elapsed_time}")
    # uptake.model.utils.delete_pkl_files(testcase_id, folder)
    # print('deleted intermediat
    # e .pkl files')


def run_parallel_routine_to_get_feq_given_mechanics_and_particle(testcase_id, zip_generator, CPUs):
    """
    Args:
        testcase_id : string - name of the investigated testcase
        mechanics_generator : generator - generator of the mechanics classes set to test
        CPUs : int - amount of CPUs used to paralellize the tasks
    Returns:
        Nothing
        runs the calculations in parallel with CPUs computation cores and
        generates the file containing the phase proportions from the phase diagrams
    """
    print(testcase_id, "STARTED")
    print(f"computing on {CPUs} CPUs")
    start = time.perf_counter()
    func = partial(compute_energy_and_get_phase_for_given_mechanics_and_particle, testcase_id)
    parallelize_energy_computation_and_phase_determination_with_mechanics_and_particle(func, zip_generator, CPUs)
    print(testcase_id, " : GENERATED ALL PKL FILES")
    generate_and_export_feq(testcase_id, folder=testcase_id)
    print(testcase_id, "GENERATED DATAFILE FEQ")
    end = time.perf_counter()
    elapsed = end - start
    print(testcase_id, "DONE")
    elapsed_time = timedelta(seconds=elapsed)
    print(f"elapsed time: {elapsed_time}")
    # uptake.model.utils.delete_pkl_files(testcase_id, folder)
    # print('deleted intermediat
    # e .pkl files')


def run_parallel_routine_to_get_feq_given_mechanics_and_particle_allvar(testcase_id, zip_generator, CPUs):
    """
    Args:
        testcase_id : string - name of the investigated testcase
        mechanics_generator : generator - generator of the mechanics classes set to test
        CPUs : int - amount of CPUs used to paralellize the tasks
    Returns:
        Nothing
        runs the calculations in parallel with CPUs computation cores and
        generates the file containing the phase proportions from the phase diagrams
    """
    print(testcase_id, "STARTED")
    print(f"computing on {CPUs} CPUs")
    start = time.perf_counter()
    func = partial(compute_energy_and_get_phase_for_given_mechanics_and_particle, testcase_id)
    parallelize_energy_computation_and_phase_determination_with_mechanics_and_particle(func, zip_generator, CPUs)
    print(testcase_id, " : GENERATED ALL PKL FILES")
    generate_and_export_feq_allvar(testcase_id, folder=testcase_id)
    print(testcase_id, "GENERATED DATAFILE FEQ")
    end = time.perf_counter()
    elapsed = end - start
    print(testcase_id, "DONE")
    elapsed_time = timedelta(seconds=elapsed)
    print(f"elapsed time: {elapsed_time}")
    # uptake.model.utils.delete_pkl_files(testcase_id, folder)
    # print('deleted intermediat
    # e .pkl files')


def launch_routine(expected_sample_size, testcase_id, CPUs):
    zip_mechanics_particle_generators = datadef.params_tester_qMC_cstsigma(expected_sample_size, CPUs, testcase_id)
    run_parallel_routine_to_get_phases_given_mechanics_and_particle(
        testcase_id, zip_mechanics_particle_generators, CPUs
    )


def launch_routine_new_settings(expected_sample_size, testcase_id, CPUs):
    zip_mechanics_particle_generators = datadef.params_qMC_mechanoadaptation_circular_new_settings(
        expected_sample_size, CPUs, testcase_id
    )
    run_parallel_routine_to_get_phases_given_mechanics_and_particle(
        testcase_id, zip_mechanics_particle_generators, CPUs
    )


def launch_routine_feq(expected_sample_size, testcase_id, CPUs):
    zip_mechanics_particle_generators = datadef.params_datafile_gamma_sigmarbar_feq(
        expected_sample_size, CPUs, testcase_id
    )
    # for zip_i in zip_mechanics_particle_generators:
    #     compute_energy_and_get_phase_for_given_mechanics_and_particle(testcase_id, zip_i)
    run_parallel_routine_to_get_feq_given_mechanics_and_particle(testcase_id, zip_mechanics_particle_generators, CPUs)


def launch_routine_feq_allvar(expected_sample_size, testcase_id, CPUs):
    zip_mechanics_particle_generators = datadef.params_datafile_gammavar_sigmarbar_rbar_feq(
        expected_sample_size, CPUs, testcase_id
    )
    run_parallel_routine_to_get_feq_given_mechanics_and_particle_allvar(
        testcase_id, zip_mechanics_particle_generators, CPUs
    )


def launch_routine_feq_allvar_new_settings(expected_sample_size, testcase_id, CPUs):
    zip_mechanics_particle_generators = datadef.params_datafile_gammavar_sigmarbar_rbar_feq_new_settings(
        expected_sample_size, CPUs, testcase_id
    )
    run_parallel_routine_to_get_feq_given_mechanics_and_particle_allvar(
        testcase_id, zip_mechanics_particle_generators, CPUs
    )


def launch_routine_feq_allvar_circular(expected_sample_size, testcase_id, CPUs):
    zip_mechanics_particle_generators = datadef.params_datafile_gammavar_sigmarbar_circular_feq(
        expected_sample_size, CPUs, testcase_id
    )
    # for zip_i in zip_mechanics_particle_generators:
    #     compute_energy_and_get_phase_for_given_mechanics_and_particle(testcase_id, zip_i)
    run_parallel_routine_to_get_feq_given_mechanics_and_particle_allvar(
        testcase_id, zip_mechanics_particle_generators, CPUs
    )


if __name__ == "__main__":

    testcase_id = "variable_gamma_elliptic_"
    expected_sample_size = 1

    CPUs = min(os.cpu_count() - 4, 75)
    # launch_routine(expected_sample_size, testcase_id, CPUs)

    # testcase_id_feq_allvar = "variable_allvar_feq_5000_"
    # expected_sample_size_feq = 5000

    # launch_routine_feq_allvar(expected_sample_size_feq, testcase_id_feq_allvar, CPUs)

    testcase_id_feq_allvar_circular = "variable_allvar_circular_feq_5000_"
    expected_sample_size_feq = 5000

    launch_routine_feq_allvar_circular(expected_sample_size_feq, testcase_id_feq_allvar_circular, CPUs)
