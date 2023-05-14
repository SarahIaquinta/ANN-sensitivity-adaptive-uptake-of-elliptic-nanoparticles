from pathlib import Path
import time
import numpy as np
import openturns as ot
import seaborn as sns
import os
import pickle

import uptake.multiprocessing.run_studies_servor as umps

def build_initial_dataset(expected_sample_size):
    testcase_id = "dataset_for_ANN_mechanadaptation_vs_passive_elliptic_newsettings_"
    CPUs = min(os.cpu_count() - 4, 75)
    # umps.launch_routine_feq_allvar(expected_sample_size, testcase_id, CPUs)
    umps.launch_routine_feq_allvar_new_settings(expected_sample_size, testcase_id, CPUs)

   
    



if __name__ == "__main__":
    expected_sample_size = 4096
    build_initial_dataset(expected_sample_size)
    classes_width = 0.047