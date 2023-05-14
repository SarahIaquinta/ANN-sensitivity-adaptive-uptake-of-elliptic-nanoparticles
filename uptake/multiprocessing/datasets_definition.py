import itertools
from math import pi

import numpy as np
from scipy.stats import qmc

import uptake.multiprocessing.utils as mputils
from uptake.model.system_definition import Fixed_Mechanical_Properties, MembraneGeometry, ParticleGeometry, Wrapping


def params_tester_qMC_cstsigma(expected_sample_size, CPUs, testcase_name="tester"):
    fixed_params = Fixed_Mechanical_Properties(
        gamma_bar_0_list=np.arange(1, 8, 0.5), sigma_bar_list=np.arange(0.5, 5.5, 0.25)
    )
    number_of_fixed_params = len(fixed_params.gamma_bar_0_list) * len(fixed_params.sigma_bar_list)
    sampler = qmc.Sobol(d=5, scramble=False)
    n = int(np.log(expected_sample_size) / np.log(2))
    sample_size = int(2 ** n)
    print("Sample size = ", sample_size)
    print("Amount of expected .pkl files", int(sample_size * number_of_fixed_params))
    estimated_time = mputils.estimate_computation_time(sample_size, number_of_fixed_params, CPUs)

    print(f"Expected computation time with {CPUs} CPUS: {estimated_time}")
    sample = sampler.random_base2(m=n)  # defining sample size = 4096
    var1 = np.array([sample[i][0] for i in range(len(sample))])
    var2 = np.array([sample[i][1] for i in range(len(sample))])
    var3 = np.array([sample[i][2] for i in range(len(sample))])
    var4 = np.array([sample[i][3] for i in range(len(sample))])
    gamma_bar_r_max = 6
    gamma_bar_r_min = 1
    gamma_bar_fs_max = 0.45
    gamma_bar_fs_min = -0.45
    gamma_bar_lambda_max = 10
    gamma_bar_lambda_min = 100

    r_bar_max = 6
    r_bar_min = 1 / 6
    gamma_bar_r_list = [i * (gamma_bar_r_max - gamma_bar_r_min) + gamma_bar_r_min for i in var1]
    gamma_bar_lambda_list = [i * (gamma_bar_lambda_max - gamma_bar_lambda_min) + gamma_bar_lambda_min for i in var2]
    gamma_bar_fs_list = [i * (gamma_bar_fs_max - gamma_bar_fs_min) + gamma_bar_fs_min for i in var3]
    r_bar_list_vertical = [i * (1 - r_bar_min) + r_bar_min for i in var4[0 : int(len(var4) / 2)]]
    r_bar_list_horizontal = [i * (r_bar_max - 1) + 1 for i in var4[int(len(var4) / 2) :]]
    r_bar_list = r_bar_list_vertical + r_bar_list_horizontal

    gamma_bar_0_list_iter_product = list(
        itertools.chain.from_iterable(
            itertools.repeat(i, len(fixed_params.sigma_bar_list)) for i in fixed_params.gamma_bar_0_list
        )
    )
    sigma_bar_list_iter_product = list(fixed_params.sigma_bar_list) * len(fixed_params.gamma_bar_0_list)

    testcase_list_repeated = itertools.chain.from_iterable(
        itertools.repeat(i, number_of_fixed_params) for i in range(int(sample_size))
    )

    gamma_bar_r_list_repeated = list(
        itertools.chain.from_iterable(
            itertools.repeat(gamma_bar_r_list[i], number_of_fixed_params) for i in range(int(sample_size))
        )
    )
    gamma_bar_fs_list_repeated = list(
        itertools.chain.from_iterable(
            itertools.repeat(gamma_bar_fs_list[i], number_of_fixed_params) for i in range(int(sample_size))
        )
    )
    gamma_bar_lambda_list_repeated = list(
        itertools.chain.from_iterable(
            itertools.repeat(gamma_bar_lambda_list[i], number_of_fixed_params) for i in range(int(sample_size))
        )
    )

    r_bar_list_repeated = list(
        itertools.chain.from_iterable(
            itertools.repeat(r_bar_list[i], number_of_fixed_params) for i in range(int(sample_size))
        )
    )

    gamma_bar_0_list_repeated = gamma_bar_0_list_iter_product * sample_size
    sigma_bar_list_repeated = sigma_bar_list_iter_product * sample_size

    testcase_number_generator = iter(testcase_list_repeated)
    generator = list(
        itertools.zip_longest(
            gamma_bar_r_list_repeated,
            gamma_bar_fs_list_repeated,
            gamma_bar_lambda_list_repeated,
            gamma_bar_0_list_repeated,
            sigma_bar_list_repeated,
        )
    )

    r_bar_generator = list(
        itertools.zip_longest(
            r_bar_list_repeated,
        )
    )

    mechanics_generator = mputils.generate_MechanicalProperties_Adaptation_classes(
        testcase_name, testcase_number_generator, generator
    )
    particle_generator = mputils.generate_Particle_classes(r_bar_generator)
    mechanics_particle_zip = itertools.zip_longest(mechanics_generator, particle_generator)
    return mechanics_particle_zip


def params_datafile_gamma_sigmarbar_feq(expected_sample_size, CPUs, testcase_name):
    sampler = qmc.Sobol(d=3, scramble=False)
    n = int(np.log(expected_sample_size) / np.log(2))
    sample_size = int(2 ** n)
    print("Sample size = ", sample_size)
    print("Amount of expected .pkl files", int(sample_size))
    estimated_time = mputils.estimate_computation_time(sample_size, 1, CPUs)

    print(f"Expected computation time with {CPUs} CPUS: {estimated_time}")
    sample = sampler.random_base2(m=n)  # defining sample size = 4096
    var1 = np.array([sample[i][0] for i in range(len(sample))])
    var2 = np.array([sample[i][1] for i in range(len(sample))])
    var3 = np.array([sample[i][2] for i in range(len(sample))])
    gamma_bar_0_max = 8
    gamma_bar_0_min = 1
    sigma_min = 0.5
    sigma_max = 5.5

    r_bar_max = 6
    r_bar_min = 1 / 6
    gamma_bar_0_list = [i * (gamma_bar_0_max - gamma_bar_0_min) + gamma_bar_0_min for i in var1]
    sigma_bar_list = [i * (sigma_max - sigma_min) + sigma_min for i in var2]
    r_bar_list_vertical = [i * (1 - r_bar_min) + r_bar_min for i in var3[0 : int(len(var3) / 2)]]
    r_bar_list_horizontal = [i * (r_bar_max - 1) + 1 for i in var3[int(len(var3) / 2) :]]
    r_bar_list = r_bar_list_vertical + r_bar_list_horizontal

    testcase_list_repeated = itertools.chain.from_iterable(
        itertools.repeat(i, sample_size) for i in range(int(sample_size))
    )

    gamma_bar_0_list_repeated = gamma_bar_0_list
    sigma_bar_list_repeated = sigma_bar_list
    r_bar_list_repeated = r_bar_list
    gamma_bar_r_list_repeated = [1] * sample_size
    gamma_bar_fs_list_repeated = [0] * sample_size
    gamma_bar_lambda_list_repeated = [50] * sample_size

    testcase_number_generator = iter(testcase_list_repeated)
    generator = list(
        itertools.zip_longest(
            gamma_bar_r_list_repeated,
            gamma_bar_fs_list_repeated,
            gamma_bar_lambda_list_repeated,
            gamma_bar_0_list_repeated,
            sigma_bar_list_repeated,
        )
    )

    r_bar_generator = list(
        itertools.zip_longest(
            r_bar_list_repeated,
        )
    )

    mechanics_generator = mputils.generate_MechanicalProperties_Adaptation_classes(
        testcase_name, testcase_number_generator, generator
    )
    particle_generator = mputils.generate_Particle_classes(r_bar_generator)
    mechanics_particle_zip = itertools.zip_longest(mechanics_generator, particle_generator)
    return mechanics_particle_zip


def params_datafile_gammavar_sigmarbar_rbar_feq(expected_sample_size, CPUs, testcase_name):
    sampler = qmc.Sobol(d=6, scramble=False)
    n = int(np.log(expected_sample_size) / np.log(2))
    sample_size = int(2 ** n)
    print("Sample size = ", sample_size)
    print("Amount of expected .pkl files", int(sample_size))
    estimated_time = mputils.estimate_computation_time(sample_size, 1, CPUs)

    print(f"Expected computation time with {CPUs} CPUS: {estimated_time}")
    sample = sampler.random_base2(m=n)  # defining sample size = 4096
    var1 = np.array([sample[i][0] for i in range(len(sample))])
    var2 = np.array([sample[i][1] for i in range(len(sample))])
    var3 = np.array([sample[i][2] for i in range(len(sample))])
    var4 = np.array([sample[i][3] for i in range(len(sample))])
    var5 = np.array([sample[i][4] for i in range(len(sample))])
    var6 = np.array([sample[i][5] for i in range(len(sample))])
    gamma_bar_0_max = 8
    gamma_bar_0_min = 1
    sigma_min = 0.5
    sigma_max = 5.5
    gamma_bar_lambda_min = 10
    gamma_bar_lambda_max = 100
    gamma_bar_r_min = 1
    gamma_bar_r_max = 6
    gamma_bar_fs_min = -0.45
    gamma_bar_fs_max = 0.45

    r_bar_max = 6
    r_bar_min = 1 / 6
    gamma_bar_0_list = [i * (gamma_bar_0_max - gamma_bar_0_min) + gamma_bar_0_min for i in var1]
    sigma_bar_list = [i * (sigma_max - sigma_min) + sigma_min for i in var2]
    r_bar_list_vertical = [i * (1 - r_bar_min) + r_bar_min for i in var3[0 : int(len(var3) / 2)]]
    r_bar_list_horizontal = [i * (r_bar_max - 1) + 1 for i in var3[int(len(var3) / 2) :]]
    r_bar_list = r_bar_list_vertical + r_bar_list_horizontal
    gamma_bar_r_list = [i * (gamma_bar_r_max - gamma_bar_r_min) + gamma_bar_r_min for i in var4]
    gamma_bar_fs_list = [i * (gamma_bar_fs_max - gamma_bar_fs_min) + gamma_bar_fs_min for i in var5]
    gamma_bar_lambda_list = [i * (gamma_bar_lambda_max - gamma_bar_lambda_min) + gamma_bar_lambda_min for i in var6]

    testcase_list_repeated = itertools.chain.from_iterable(
        itertools.repeat(i, sample_size) for i in range(int(sample_size))
    )

    gamma_bar_0_list_repeated = gamma_bar_0_list
    sigma_bar_list_repeated = sigma_bar_list
    r_bar_list_repeated = r_bar_list
    gamma_bar_r_list_repeated = gamma_bar_r_list
    gamma_bar_fs_list_repeated = gamma_bar_fs_list
    gamma_bar_lambda_list_repeated = gamma_bar_lambda_list

    testcase_number_generator = iter(testcase_list_repeated)
    generator = list(
        itertools.zip_longest(
            gamma_bar_r_list_repeated,
            gamma_bar_fs_list_repeated,
            gamma_bar_lambda_list_repeated,
            gamma_bar_0_list_repeated,
            sigma_bar_list_repeated,
        )
    )

    r_bar_generator = list(
        itertools.zip_longest(
            r_bar_list_repeated,
        )
    )

    mechanics_generator = mputils.generate_MechanicalProperties_Adaptation_classes(
        testcase_name, testcase_number_generator, generator
    )
    particle_generator = mputils.generate_Particle_classes(r_bar_generator)
    mechanics_particle_zip = itertools.zip_longest(mechanics_generator, particle_generator)
    return mechanics_particle_zip


def params_datafile_gammavar_sigmarbar_rbar_feq_new_settings(expected_sample_size, CPUs, testcase_name):
    sampler = qmc.Sobol(d=6, scramble=False)
    n = int(np.log(expected_sample_size) / np.log(2))
    sample_size = int(2 ** n)
    print("Sample size = ", sample_size)
    print("Amount of expected .pkl files", int(sample_size))
    estimated_time = mputils.estimate_computation_time(sample_size, 1, CPUs)

    print(f"Expected computation time with {CPUs} CPUS: {estimated_time}")
    sample = sampler.random_base2(m=n)  # defining sample size = 4096
    var1 = np.array([sample[i][0] for i in range(len(sample))])
    var2 = np.array([sample[i][1] for i in range(len(sample))])
    var3 = np.array([sample[i][2] for i in range(len(sample))])
    var4 = np.array([sample[i][3] for i in range(len(sample))])
    var5 = np.array([sample[i][4] for i in range(len(sample))])
    var6 = np.array([sample[i][5] for i in range(len(sample))])
    gamma_bar_0_max = 8
    gamma_bar_0_min = 1
    sigma_min = 0.5
    sigma_max = 5.5
    gamma_bar_lambda_min = 10
    gamma_bar_lambda_max = 50
    gamma_bar_r_min = 1
    gamma_bar_r_max = 6
    gamma_bar_fs_min = -0.45
    gamma_bar_fs_max = 0.45

    r_bar_max = 6
    r_bar_min = 1
    gamma_bar_0_list = [i * (gamma_bar_0_max - gamma_bar_0_min) + gamma_bar_0_min for i in var1]
    sigma_bar_list = [i * (sigma_max - sigma_min) + sigma_min for i in var2]
    r_bar_list_horizontal = [i * (r_bar_max - r_bar_min) + 1 for i in var3[int(len(var3) / 2) :]]
    r_bar_list_vertical = [1 / i for i in r_bar_list_horizontal]
    r_bar_list = r_bar_list_vertical + r_bar_list_horizontal
    gamma_bar_r_list = [i * (gamma_bar_r_max - gamma_bar_r_min) + gamma_bar_r_min for i in var4]
    gamma_bar_fs_list = [i * (gamma_bar_fs_max - gamma_bar_fs_min) + gamma_bar_fs_min for i in var5]
    gamma_bar_lambda_list = [i * (gamma_bar_lambda_max - gamma_bar_lambda_min) + gamma_bar_lambda_min for i in var6]

    testcase_list_repeated = itertools.chain.from_iterable(
        itertools.repeat(i, sample_size) for i in range(int(sample_size))
    )

    gamma_bar_0_list_repeated = gamma_bar_0_list
    sigma_bar_list_repeated = sigma_bar_list
    r_bar_list_repeated = r_bar_list
    gamma_bar_r_list_repeated = gamma_bar_r_list
    gamma_bar_fs_list_repeated = gamma_bar_fs_list
    gamma_bar_lambda_list_repeated = gamma_bar_lambda_list

    testcase_number_generator = iter(testcase_list_repeated)
    generator = list(
        itertools.zip_longest(
            gamma_bar_r_list_repeated,
            gamma_bar_fs_list_repeated,
            gamma_bar_lambda_list_repeated,
            gamma_bar_0_list_repeated,
            sigma_bar_list_repeated,
        )
    )

    r_bar_generator = list(
        itertools.zip_longest(
            r_bar_list_repeated,
        )
    )

    mechanics_generator = mputils.generate_MechanicalProperties_Adaptation_classes(
        testcase_name, testcase_number_generator, generator
    )
    particle_generator = mputils.generate_Particle_classes(r_bar_generator)
    mechanics_particle_zip = itertools.zip_longest(mechanics_generator, particle_generator)
    return mechanics_particle_zip


def params_datafile_gammavar_sigmarbar_circular_feq(expected_sample_size, CPUs, testcase_name):
    sampler = qmc.Sobol(d=5, scramble=False)
    n = int(np.log(expected_sample_size) / np.log(2))
    sample_size = int(2 ** n)
    print("Sample size = ", sample_size)
    print("Amount of expected .pkl files", int(sample_size))
    estimated_time = mputils.estimate_computation_time(sample_size, 1, CPUs)

    print(f"Expected computation time with {CPUs} CPUS: {estimated_time}")
    sample = sampler.random_base2(m=n)  # defining sample size = 4096
    var1 = np.array([sample[i][0] for i in range(len(sample))])
    var2 = np.array([sample[i][1] for i in range(len(sample))])
    var3 = np.array([sample[i][2] for i in range(len(sample))])
    var4 = np.array([sample[i][3] for i in range(len(sample))])
    var5 = np.array([sample[i][4] for i in range(len(sample))])
    gamma_bar_0_max = 8
    gamma_bar_0_min = 1
    sigma_min = 0.5
    sigma_max = 5.5
    gamma_bar_lambda_min = 10
    gamma_bar_lambda_max = 100
    gamma_bar_r_min = 1
    gamma_bar_r_max = 6
    gamma_bar_fs_min = -0.45
    gamma_bar_fs_max = 0.45

    gamma_bar_0_list = [i * (gamma_bar_0_max - gamma_bar_0_min) + gamma_bar_0_min for i in var1]
    sigma_bar_list = [i * (sigma_max - sigma_min) + sigma_min for i in var2]
    gamma_bar_r_list = [i * (gamma_bar_r_max - gamma_bar_r_min) + gamma_bar_r_min for i in var3]
    gamma_bar_fs_list = [i * (gamma_bar_fs_max - gamma_bar_fs_min) + gamma_bar_fs_min for i in var4]
    gamma_bar_lambda_list = [i * (gamma_bar_lambda_max - gamma_bar_lambda_min) + gamma_bar_lambda_min for i in var5]

    testcase_list_repeated = itertools.chain.from_iterable(
        itertools.repeat(i, sample_size) for i in range(int(sample_size))
    )

    gamma_bar_0_list_repeated = gamma_bar_0_list
    sigma_bar_list_repeated = sigma_bar_list
    r_bar_list_repeated = [1] * sample_size
    gamma_bar_r_list_repeated = gamma_bar_r_list
    gamma_bar_fs_list_repeated = gamma_bar_fs_list
    gamma_bar_lambda_list_repeated = gamma_bar_lambda_list

    testcase_number_generator = iter(testcase_list_repeated)
    generator = list(
        itertools.zip_longest(
            gamma_bar_r_list_repeated,
            gamma_bar_fs_list_repeated,
            gamma_bar_lambda_list_repeated,
            gamma_bar_0_list_repeated,
            sigma_bar_list_repeated,
        )
    )

    r_bar_generator = list(
        itertools.zip_longest(
            r_bar_list_repeated,
        )
    )

    mechanics_generator = mputils.generate_MechanicalProperties_Adaptation_classes(
        testcase_name, testcase_number_generator, generator
    )
    particle_generator = mputils.generate_Particle_classes(r_bar_generator)
    mechanics_particle_zip = itertools.zip_longest(mechanics_generator, particle_generator)
    return mechanics_particle_zip


if __name__ == "__main__":
    particle_generator, mechanics_generator, membrane, wrapping = params_tester_qMC_cstsigma(
        expected_sample_size=3, testcase_name="testqMC"
    )
    print("done")
