import itertools
from datetime import timedelta
from math import pi

import numpy as np

from uptake.model.system_definition import MechanicalProperties_Adaptation, MembraneGeometry, ParticleGeometry, Wrapping


def generate_MechanicalProperties_Adaptation_classes(
    testcase_name, testcase_number_generator, mech_properties_argument_generator_fixed
):
    """
    Args:
        testcase_name : string - name of the testcase to identify the dataset
        testcase_number_generator : generator - range of numbers from 0 to the
                                    amout of phase diagrams generated in the
                                    dataset
        mech_properties_argument_generator_fixed : generator - range of the
                                    possible combinations of input parameters
                                    for the MechanicalProperties_Adaptation
                                    class, excepted the testcase variable
    Returns:
        mech_properties : generator - set of the possible combinations of
                                    all the input parameters for the
                                    MechanicalProperties_Adaptation class
    """
    for fixed_mech_properties_arguments in mech_properties_argument_generator_fixed:
        testcase_number = str(next(testcase_number_generator))
        testcase_final = testcase_name + testcase_number + "_"
        mech_properties = MechanicalProperties_Adaptation(testcase_final, *fixed_mech_properties_arguments)
        yield mech_properties


def generate_Particle_classes(r_bar_generator):
    """
    Args:
        r_bar_generator : generator - range of the
                                    possible r_bar
    Returns:
        particle : generator of the class Particle
    """
    for r_bar_argument in r_bar_generator:
        particle = ParticleGeometry(*r_bar_argument, 2 * pi, 300)
        yield particle



def define_global_Wrapping_class():
    wrapping = Wrapping(wrapping_list=np.arange(0.03, 0.97, 0.003125))
    return wrapping


def estimate_computation_time(sample_size, number_of_fixed_params, CPUs):
    time_for_one_sample_one_CPU = 85120 / 286720 * 75
    estimated_computation_time_second = time_for_one_sample_one_CPU * sample_size * number_of_fixed_params / CPUs
    delta = timedelta(seconds=estimated_computation_time_second)
    return delta
