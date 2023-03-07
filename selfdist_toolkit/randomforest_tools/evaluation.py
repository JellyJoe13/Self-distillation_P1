import json
import typing
import numpy as np
from tqdm import tqdm
from selfdist_toolkit.randomforest_tools.self_distillation import execute_sd_rf_test_simplified
import os
import pandas as pd

# define storage path of results
results_folder_name = "results/random_forest/experiments_check/"


def helper_merge_accuracy_dicts(
        aid: int,
        teacher: typing.Dict[str, float],
        student: typing.Dict[str, float]
) -> typing.Dict[str, float]:
    """
    Helper function that merges two accuracy dicts (used by random forest testing methods) and generates a single dict
    which can then be dumped to a file more easily or used to create a pandas DataFrame.

    Parameters
    ----------
    aid : int
        Experiment id (information to put in the dict)
    teacher : typing.Dict[str, float]
        Measured accuracies of the teacher
    student : typing.Dict[str, float]
        Measured accuracy of the student

    Returns
    -------
    out : typing.Dict[str, float]
        Merged dictionary
    """

    # initialize dict
    out = {
        "aid": aid
    }

    # put teacher information in
    for key in teacher.keys():
        out["teacher_" + key] = teacher[key]

    # put student information in
    for key in student.keys():
        out["student_" + key] = student[key]

    # return dict
    return out


def evaluate_assays_rf(
        perc_sd: float,
        mode: str = "chem-desc",
        data_folder: str = "data/"
) -> pd.DataFrame:
    """
    Function that executes the random forest testing from the function execute_sd_rf_test_simplified(...) for all aids.
    It calculates the teacher and student accuracies, temporarily stores them in a json file for when the process
    crashes to not loose progress. Puts all measured results into pandas Dataframe, which it saves in the
    results/random_forest/experiments_check/ folder and also returns it.

    Parameters
    ----------
    perc_sd : float
        Percentage of self distillation elements to generate in proportion to the size of the data for the experiment
    mode : str, optional
        Data mode, either chem-desc or fingerprint
    data_folder : str, optional
        Folder where the data is stored. Usually data/

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the accuracies measured in the function
    """

    # FOLDER RELATED WORK
    # get folder path to storage
    storage_folder = results_folder_name + mode + "/"

    # check if the folder where the data from this operation exists and if not create it
    if not os.path.exists(storage_folder):
        os.makedirs(storage_folder)

    # check if the file where it would have been stored is already present and return it.
    file_path = storage_folder + "perc-" + str(perc_sd) + ".csv"
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)

    # check if temporary folder is present and create it if not.
    temp_path = storage_folder + "tmp/"
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    # OPERATIONS INSTRUCTIONS
    # load assay ids to traverse
    aid_list = np.load(data_folder + "experiment-wise/ToC.npy")

    # load which of the experiments are already present in the temp folder (mechanism for aborted function to not loose
    # progress)
    file_beginning = "perc-" + str(perc_sd) + "_"
    already_computed = []
    for file in os.listdir(temp_path):
        # check if the rigth percentage
        if not file[:len(file_beginning)] == file_beginning:
            continue
        elif not file[-5:] == ".json":
            continue
        else:
            already_computed.append(int(file[len(file_beginning):-5]))

    # initialize pandas dataframe to put the data into
    df = pd.DataFrame()

    # iterate over aids
    for aid in tqdm(aid_list):
        # check if experiment has minimum requirements:

        # initialize accuracy storages
        teacher_acc_dict = None
        student_acc_dict = None

        # temp file path
        temp_file_path = temp_path + "desc-" + str(perc_sd) + "_" + str(aid) + ".json"

        # if the experiment has already been tested, load it instead
        if aid in already_computed:
            # load data from file
            with open(temp_file_path, "r") as f:
                temp = json.load(f)
                teacher_acc_dict = temp["teacher"]
                student_acc_dict = temp["student"]
        else:
            try:
                # calculate scores
                teacher_acc_dict, student_acc_dict = execute_sd_rf_test_simplified(
                    aid=aid,
                    perc_sd=perc_sd,
                    mode=mode
                )
                # save it in case computation is interrupted
                with open(temp_file_path, "w") as f:
                    # write results to storage
                    json.dump(
                        {
                            "teacher": teacher_acc_dict,
                            "student": student_acc_dict
                        },
                        f,
                        indent=4
                    )
            except ValueError:
                continue

        # add file data to the overall pandas dataframe
        df = df.append(
            helper_merge_accuracy_dicts(
                aid=aid,
                teacher=teacher_acc_dict,
                student=student_acc_dict
            ),
            ignore_index=True
        )

    # store the generated pandas dataframe to file name (and remove temporary files)
    # store result
    df.to_csv(file_path)

    # remove temp files
    for aid in aid_list:
        # determine file name (if it happens to exist)
        theoretical_path = temp_path + "desc-" + str(perc_sd) + "_" + str(aid) + ".json"

        # if present delete it
        if os.path.isfile(theoretical_path):
            os.remove(theoretical_path)

    # return the resulting DataFrame
    return df
