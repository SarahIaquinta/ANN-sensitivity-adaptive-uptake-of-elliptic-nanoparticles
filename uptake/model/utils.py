import glob
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

import numpy as np

from uptake import make_data_folder
import scipy.signal

def define_pkl_filename(particle, mechanics):
    """
    Args:
        params: class of imput parameters
    Returns:
        str: name of the output pkl file
    """
    outfile = mechanics.testcase
    outfile += "_r="
    outfile += (str)(np.round(particle.r_bar, 3))
    outfile += "_g0="
    outfile += (str)(mechanics.gamma_bar_0)
    outfile += "_gr="
    outfile += (str)(mechanics.gamma_bar_r)
    outfile += "_gfs="
    outfile += (str)(mechanics.gamma_bar_fs)
    outfile += "_gl="
    outfile += (str)(mechanics.gamma_bar_lambda)
    outfile += "_s="
    outfile += (str)(mechanics.sigma_bar)
    outfile += ".pkl"
    return outfile

def determine_eq_energy(f_list, energy_list):
    min_energy_index_list = scipy.signal.argrelmin(energy_list)
    max_energy_index_list = scipy.signal.argrelmax(energy_list)
    min_energy_index_list = list(min_energy_index_list[0])
    max_energy_index_list = list(max_energy_index_list[0])
    min_energy_index_list_initial = min_energy_index_list.copy()
    max_energy_index_list_initial = max_energy_index_list.copy()
    if len(max_energy_index_list_initial) > 0:
        for i in range(min(len(min_energy_index_list_initial), len(max_energy_index_list_initial))):
            index_min = min_energy_index_list_initial[i]
            index_max = max_energy_index_list_initial[i]
            diff_index = abs(index_min - index_max)
            close_indices = diff_index == 1
            if close_indices:  # check if the minimum is directly after of before a minimum
                # (i.e. if there is a peak of energy due to artefacts)
                min_energy_index_list.remove(index_min)
                max_energy_index_list.remove(index_max)
    # check if the minimum is reached for f_list[-1]
    if energy_list[-1] < energy_list[-2]:
        min_energy_index_list = min_energy_index_list + [-1]

    # check if the minimum is reached for f_list[0]
    if energy_list[0] < energy_list[1]:
        min_energy_index_list = [f_list[0]] + min_energy_index_list
    if len(min_energy_index_list) == 0:
        min_energy_index_list = [0]
    min_energy_list = [energy_list[int(k)] for k in min_energy_index_list]
    f_min_energy_list = [f_list[int(k)] for k in min_energy_index_list]
    if len(max_energy_index_list) > 0:
        max_energy_list = [energy_list[int(k)] for k in max_energy_index_list]
        f_max_energy_list = [f_list[int(k)] for k in max_energy_index_list]
    else:
        max_energy_list = []
        f_max_energy_list = []
    # managing possible scipy.signal.argrelextrema outuput types
    if type(min_energy_list[0]) == np.ndarray:
        min_energy_list = min_energy_list[0]
        f_min_energy_list = f_min_energy_list[0]

    f_eq = f_min_energy_list[0]
    energy_eq = min_energy_list[0]
    return f_eq, energy_eq


def determine_min_energy(f_list, energy_list):
    min_energy_index_list = scipy.signal.argrelmin(energy_list)
    max_energy_index_list = scipy.signal.argrelmax(energy_list)
    min_energy_index_list = list(min_energy_index_list[0])
    max_energy_index_list = list(max_energy_index_list[0])
    min_energy_index_list_initial = min_energy_index_list.copy()
    max_energy_index_list_initial = max_energy_index_list.copy()
    if len(max_energy_index_list_initial) > 0:
        for i in range(min(len(min_energy_index_list_initial), len(max_energy_index_list_initial))):
            index_min = min_energy_index_list_initial[i]
            index_max = max_energy_index_list_initial[i]
            diff_index = abs(index_min - index_max)
            close_indices = diff_index == 1
            if close_indices:  # check if the minimum is directly after of before a minimum
                # (i.e. if there is a peak of energy due to artefacts)
                min_energy_index_list.remove(index_min)
                max_energy_index_list.remove(index_max)
    # check if the minimum is reached for f_list[-1]
    if energy_list[-1] < energy_list[-2]:
        min_energy_index_list = min_energy_index_list + [-1]

    # check if the minimum is reached for f_list[0]
    if energy_list[0] < energy_list[1]:
        min_energy_index_list = [f_list[0]] + min_energy_index_list
    if len(min_energy_index_list) == 0:
        min_energy_index_list = [0]
    min_energy_list = [energy_list[int(k)] for k in min_energy_index_list]
    f_min_energy_list = [f_list[int(k)] for k in min_energy_index_list]
    if len(max_energy_index_list) > 0:
        max_energy_list = [energy_list[int(k)] for k in max_energy_index_list]
        f_max_energy_list = [f_list[int(k)] for k in max_energy_index_list]
    else:
        max_energy_list = []
        f_max_energy_list = []
    # managing possible scipy.signal.argrelextrema outuput types
    if type(min_energy_list[0]) == np.ndarray:
        min_energy_list = min_energy_list[0]
        f_min_energy_list = f_min_energy_list[0]

    f_eq = f_min_energy_list[0]
    energy_eq = min_energy_list[0]
    return f_min_energy_list, min_energy_list




def delete_empty_pkl_files():
    path = Path.cwd()
    path_extension = "*.pkl"
    file_path = path / path_extension
    deleted_file_counter = 0
    for filename in glob.glob(file_path):
        file_size = os.path.getsize(filename)
        if file_size < 1:
            os.remove(filename)
            deleted_file_counter += 1
    # print('deleted %d empty .pkl files' %deleted_file_counter)


def delete_pkl_files(testcase, folder):
    path = make_data_folder(folder)
    path_extension = testcase + "*.pkl"
    for filename in Path(".").glob(path_extension):
        filename.unlink(missing_ok=True)


# def get_folder_size(path):
#     """
#     A function to compute the size of the current folder, in bytes

#     Attributes:
#         ----------
#         path: str
#             path of the folder

#     Returns:
#         -------
#         total_size: float
#             size of the current folder, in bytes
#     """
#     total_size = 0
#     for dirpath, _, filenames in os.walk(path):
#         for f in filenames:
#             fp = os.path.join(dirpath, f)
#             # skip if it is symbolic link
#             if not os.path.islink(fp):
#                 total_size += os.path.getsize(fp)
#     return total_size
