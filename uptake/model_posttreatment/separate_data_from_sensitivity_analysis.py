import pickle

import numpy as np
from tqdm import tqdm

from uptake.model_posttreatment.phase_analysis import Files


def get_list_of_testcase_id_in_folder(testcase, folder):
    files = Files(testcase)
    files_from_sensitivity_analysis = files.import_files(folder)
    indexofunderscore_list = [
        len(testcase) + L[len(testcase) : len(testcase) + 10].index("_") for L in files_from_sensitivity_analysis
    ]
    list_of_testcases_id = {
        files_from_sensitivity_analysis[i][0 : indexofunderscore_list[i]] for i in range(len(indexofunderscore_list))
    }
    return list_of_testcases_id


# def extract_phase_proportions_for_X0(testcase, folder):
#     """
#         - Reads the files belonging to the "testcase" dataset
#         - Among theses, separates the one belonging to a same phase diagram
#         - Get the phase proportions
#         - Gathers the X0 vector with the corresponding phase proportions in an array
#         - Saves the array into a .pkl file and a .txt file
#     Args:
#         testcase : string - name of the testcase to identify the dataset
#     Returns:
#         Nothing
#     """
#     files = Files(testcase)
#     files_from_sensitivity_analysis = files.import_files(folder)
#     amount_of_files = len(files_from_sensitivity_analysis)
#     print(amount_of_files)
#     list_of_testcases_id = list(get_list_of_testcase_id_in_folder(testcase))
#     amount_of_testcases = len(list_of_testcases_id)
#     parameters_data = np.zeros((amount_of_testcases, 7))
#     filename = "data_for_sensitivity_analysis_" + testcase + ".txt"
#     filename_pkl = "data_for_sensitivity_analysis_" + testcase + ".pkl"
#     f = open(filename, "w")
#     f.write("Id \t  gr \t gf \t gl \t prop phase 1 \t prop phase 2 \t prop phase 3 \n")
#     for i in tqdm(range(amount_of_testcases)):
#         testcase_id = list_of_testcases_id[i] + "_"
#         # print(testcase_id)
#         parameters_data[i, 0] = i
#         files_testcase = Files(testcase_id)
#         files_testcase_list = files_testcase.import_files(folder)
#         file_phase = [0, 0, 0]
#         for j in range(len(files_testcase_list)):
#             outfile = files_testcase_list[j]
#             _, _, _, _, _, _, _, _, wrapping_phase_number = files.extract_data_from_file(outfile)
#             file_phase[int(wrapping_phase_number) - 1] += 1
#         file_phase = file_phase / np.sum(file_phase)  # computes phase proportions
#         _, mechanics, _, _, _, _, _, _, wrapping_phase_number = files.extract_data_from_file(files_testcase_list[0])
#         parameters_data[i, -3], parameters_data[i, -2], parameters_data[i, -1] = (
#             file_phase[-3],
#             file_phase[-2],
#             file_phase[-1],
#         )
#         parameters_data[i, 1] = mechanics.gamma_bar_r
#         parameters_data[i, 2] = mechanics.gamma_bar_fs
#         parameters_data[i, 3] = mechanics.gamma_bar_lambda
#         f.write(
#             testcase_id
#             + "\t"
#             + str(parameters_data[i, 1])
#             + "\t"
#             + str(parameters_data[i, 2])
#             + "\t"
#             + str(parameters_data[i, 3])
#             + "\t"
#             + str(parameters_data[i, 4])
#             + "\t"
#             + str(parameters_data[i, 5])
#             + "\t"
#             + str(parameters_data[i, 6])
#             + "\n"
#         )
#     f.close()
#     with open(filename_pkl, "wb") as f:
#         pickle.dump(parameters_data, f)


def generate_and_export_phase_proportions(testcase, folder):
    """
        - Reads the files belonging to the "testcase" dataset
        - Among theses, separates the one belonging to a same phase diagram
        - Get the phase proportions
        - Gathers the X0 vector with the corresponding phase proportions in an array
        - Saves the array into a .pkl file and a .txt file
    Args:
        testcase : string - name of the testcase to identify the dataset
    Returns:
        Nothing
    """
    files = Files(testcase)
    files_from_sensitivity_analysis = files.import_files(folder)
    amount_of_files = len(files_from_sensitivity_analysis)
    print("amount of post-treated files: ", amount_of_files)
    list_of_testcases_id = list(get_list_of_testcase_id_in_folder(testcase, folder))
    amount_of_testcases = len(list_of_testcases_id)
    parameters_data = np.zeros((amount_of_testcases, 8))
    filename = "data_for_sensitivity_analysis_" + testcase + ".txt"
    filename_pkl = "data_for_sensitivity_analysis_" + testcase + ".pkl"
    f = open(filename, "w")
    f.write("Id \t gr \t gf \t gl \t r \t prop phase 1 \t prop phase 2 \t prop phase 3 \n")
    for i in tqdm(range(amount_of_testcases)):
        testcase_id = list_of_testcases_id[i] + "_"
        parameters_data[i, 0] = i
        files_testcase = Files(testcase_id)
        files_testcase_list = files_testcase.import_files(folder)
        file_phase = [0, 0, 0]
        for j in range(len(files_testcase_list)):
            outfile = files_testcase_list[j]
            _, _, _, _, _, _, _, _, wrapping_phase_number = files.extract_data_from_file(outfile, folder)
            file_phase[int(wrapping_phase_number) - 1] += 1
        file_phase = file_phase / np.sum(file_phase)  # computes phase proportions
        particle, mechanics, _, _, _, _, _, _, wrapping_phase_number = files.extract_data_from_file(
            files_testcase_list[0], folder
        )
        parameters_data[i, -3], parameters_data[i, -2], parameters_data[i, -1] = (
            file_phase[-3],
            file_phase[-2],
            file_phase[-1],
        )
        parameters_data[i, 1] = mechanics.gamma_bar_r
        parameters_data[i, 2] = mechanics.gamma_bar_fs
        parameters_data[i, 3] = mechanics.gamma_bar_lambda
        parameters_data[i, 4] = particle.r_bar
        f.write(
            testcase_id
            + "\t"
            + str(parameters_data[i, 1])
            + "\t"
            + str(parameters_data[i, 2])
            + "\t"
            + str(parameters_data[i, 3])
            + "\t"
            + str(parameters_data[i, 4])
            + "\t"
            + str(parameters_data[i, 5])
            + "\t"
            + str(parameters_data[i, 6])
            + "\t"
            + str(parameters_data[i, 7])
            + "\n"
        )
    f.close()
    with open(filename_pkl, "wb") as f:
        pickle.dump(parameters_data, f)


def generate_and_export_feq(testcase, folder):
    """
        - Reads the files belonging to the "testcase" dataset
        - Among theses, separates the one belonging to a same phase diagram
        - Get the phase proportions
        - Gathers the X0 vector with the corresponding phase proportions in an array
        - Saves the array into a .pkl file and a .txt file
    Args:
        testcase : string - name of the testcase to identify the dataset
    Returns:
        Nothing
    """
    files = Files(testcase)
    files_from_sensitivity_analysis_feq = files.import_files(folder)
    amount_of_files = len(files_from_sensitivity_analysis_feq)
    print("amount of post-treated files: ", amount_of_files)
    list_of_testcases_id = list(get_list_of_testcase_id_in_folder(testcase, folder))
    amount_of_testcases = len(list_of_testcases_id)
    parameters_data = np.zeros((amount_of_files, 6))
    filename = "data_for_sensitivity_analysis_feq_" + testcase + ".txt"
    filename_pkl = "data_for_sensitivity_analysis_feq_" + testcase + ".pkl"
    f = open(filename, "w")
    f.write("Id \t go \t s \t r \t feq \t phasis \n")
    for i in tqdm(range(amount_of_files)):
        outfile = files_from_sensitivity_analysis_feq[i]
        particle, mechanics, _, _, _, _, f_eq, _, wrapping_phase_number = files.extract_data_from_file(outfile, folder)
        parameters_data[i, 0] = i + 1
        parameters_data[i, 1] = mechanics.gamma_bar_0
        parameters_data[i, 2] = mechanics.sigma_bar
        parameters_data[i, 3] = particle.r_bar
        parameters_data[i, 4] = f_eq
        parameters_data[i, 5] = wrapping_phase_number
        f.write(
            str(i + 1)
            + "\t"
            + str(parameters_data[i, 1])
            + "\t"
            + str(parameters_data[i, 2])
            + "\t"
            + str(parameters_data[i, 3])
            + "\t"
            + str(parameters_data[i, 4])
            + "\t"
            + str(parameters_data[i, 5])
            + "\n"
        )
    f.close()
    with open(filename_pkl, "wb") as f:
        pickle.dump(parameters_data, f)


def generate_and_export_feq_allvar(testcase, folder):
    """
        - Reads the files belonging to the "testcase" dataset
        - Among theses, separates the one belonging to a same phase diagram
        - Get the phase proportions
        - Gathers the X0 vector with the corresponding phase proportions in an array
        - Saves the array into a .pkl file and a .txt file
    Args:
        testcase : string - name of the testcase to identify the dataset
    Returns:
        Nothing
    """
    files = Files(testcase)
    files_from_sensitivity_analysis_feq = files.import_files(folder)
    amount_of_files = len(files_from_sensitivity_analysis_feq)
    print("amount of post-treated files: ", amount_of_files)
    list_of_testcases_id = list(get_list_of_testcase_id_in_folder(testcase, folder))
    amount_of_testcases = len(list_of_testcases_id)
    parameters_data = np.zeros((amount_of_files, 9))
    filename = "data_for_sensitivity_analysis_feq_" + testcase + ".txt"
    filename_pkl = "data_for_sensitivity_analysis_feq_" + testcase + ".pkl"
    f = open(filename, "w")
    f.write("Id \t go \t s \t gr \t gf \t gl\t r \t feq \t phasis \n")
    for i in tqdm(range(amount_of_files)):
        outfile = files_from_sensitivity_analysis_feq[i]
        particle, mechanics, _, _, _, _, f_eq, _, wrapping_phase_number = files.extract_data_from_file(outfile, folder)
        parameters_data[i, 0] = i + 1
        parameters_data[i, 1] = mechanics.gamma_bar_0
        parameters_data[i, 2] = mechanics.sigma_bar
        parameters_data[i, 3] = mechanics.gamma_bar_r
        parameters_data[i, 4] = mechanics.gamma_bar_fs
        parameters_data[i, 5] = mechanics.gamma_bar_lambda
        parameters_data[i, 6] = particle.r_bar
        parameters_data[i, 7] = f_eq
        parameters_data[i, 8] = wrapping_phase_number
        f.write(
            str(i + 1)
            + "\t"
            + str(parameters_data[i, 1])
            + "\t"
            + str(parameters_data[i, 2])
            + "\t"
            + str(parameters_data[i, 3])
            + "\t"
            + str(parameters_data[i, 4])
            + "\t"
            + str(parameters_data[i, 5])
            + "\t"
            + str(parameters_data[i, 6])
            + "\t"
            + str(parameters_data[i, 7])
            + "\t"
            + str(parameters_data[i, 8])
            + "\n"
        )
    f.close()
    with open(filename_pkl, "wb") as f:
        pickle.dump(parameters_data, f)


# def extract_feq_for_X0(testcase, filename, folder):
#     """
#         - Reads the files belonging to the "testcase" dataset
#         - Among theses, separates the one belonging to a same phase diagram
#         - Get the phase proportions
#         - Gathers the X0 vector with the corresponding phase proportions in an array
#         - Saves the array into a .pkl file and a .txt file
#     Args:
#         testcase : string - name of the testcase to identify the dataset
#     Returns:
#         Nothing
#     """
#     files = Files(testcase)
#     outfile_list = files.import_files(folder)
#     amount_of_files = len(outfile_list)
#     print(amount_of_files)
#     parameters_data = np.zeros((amount_of_files, 4))
#     filename = filename + ".txt"
#     filename_pkl = filename + ".pkl"
#     f = open(filename, "w")
#     f.write("Id \t g0 \t s0  \t f_eq \n")
#     for i in tqdm(range(amount_of_files)):
#         outfile = outfile_list[i]
#         parameters_data[i, 0] = i
#         _, mechanics, _, _, _, _, f_eq, _, _ = files.extract_data_from_file(outfile, folder)
#         parameters_data[i, -1] = f_eq
#         parameters_data[i, 1] = mechanics.gamma_bar_0
#         parameters_data[i, 2] = mechanics.sigma_bar_0

#         f.write(
#             str(i)
#             + "\t"
#             + str(parameters_data[i, 1])
#             + "\t"
#             + str(parameters_data[i, 2])
#             + "\t"
#             + str(parameters_data[i, 3])
#             + "\n"
#         )
#     f.close()
#     with open(filename_pkl, "wb") as f:
#         pickle.dump(parameters_data, f)


# def extract_feq_and_phase_for_X0(testcase, filename, folder):
#     """
#         - Reads the files belonging to the "testcase" dataset
#         - Among theses, separates the one belonging to a same phase diagram
#         - Get the phase proportions
#         - Gathers the X0 vector with the corresponding phase proportions in an array
#         - Saves the array into a .pkl file and a .txt file
#     Args:
#         testcase : string - name of the testcase to identify the dataset
#     Returns:
#         Nothing
#     """
#     files = Files(testcase)
#     outfile_list = files.import_files(folder)
#     amount_of_files = len(outfile_list)
#     print(amount_of_files)
#     parameters_data = np.zeros((amount_of_files, 5))
#     filename = filename + ".txt"
#     filename_pkl = filename + ".pkl"
#     f = open(filename, "w")
#     f.write("Id \t g0 \t s0  \t f_eq \t phase \n")
#     for i in tqdm(range(amount_of_files)):
#         outfile = outfile_list[i]
#         parameters_data[i, 0] = i
#         _, mechanics, _, _, _, _, f_eq, _, wrapping_phase_number = files.extract_data_from_file(outfile, folder)
#         parameters_data[i, -1] = wrapping_phase_number
#         parameters_data[i, -2] = f_eq
#         parameters_data[i, 1] = mechanics.gamma_bar_0
#         parameters_data[i, 2] = mechanics.sigma_bar_0

#         f.write(
#             str(i)
#             + "\t"
#             + str(parameters_data[i, 1])
#             + "\t"
#             + str(parameters_data[i, 2])
#             + "\t"
#             + str(parameters_data[i, 3])
#             + "\t"
#             + str(parameters_data[i, 4])
#             + "\n"
#         )
#     f.close()
#     with open(filename_pkl, "wb") as f:
#         pickle.dump(parameters_data, f)


if __name__ == "__main__":
    print("separate_data_from_sensitivity_analysis.py")
