import typing
import numpy as np
from tqdm import tqdm
from selfdist_toolkit.randomforest_tools.self_distillation import execute_sd_rf_test_simplified


def evaluate_assays_rf(
        perc_sd: float,
        mode: str = "chem-desc",
        data_folder: str = "data/"
) -> typing.Tuple[float, float]:
    # load assay ids to traverse
    aid_list = np.load(data_folder + "experiment-wise/ToC.npy")

    # iterate over aids
    for aid in tqdm(aid_list):
        teacher_acc_dict, student_acc_dict = execute_sd_rf_test_simplified(
            aid=aid,
            perc_sd=perc_sd,
            mode=mode
        )
