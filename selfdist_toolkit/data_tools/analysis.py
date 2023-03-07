import multiprocess
import os.path
import typing
import numpy as np
import pandas as pd
from tqdm import tqdm
from selfdist_toolkit.data_tools import loading


def is_experiment_good(
        aid: int,
        min_number_molecules: int,
        min_number_positive_percentage: float,
        contains_true: bool,
        path_data: str = "data/"
) -> bool:
    """
    Function that determines whether an experiment can be considered good in the sense of it fulfilling a few
    requirements:
    - have at least <param> molecules tested
    - have at least <param controlled> active molecules
    - have at least a <param controlled> percentage of active molecules

    Parameters
    ----------
    aid : int
        Experiment to test
    min_number_molecules : int
        Number of minimum molecules an experiment needs to be considered good
    min_number_positive_percentage : float
        Percentage of positive molecules an experiments needs to be considered good if the parameter contains_true is
        set to true
    contains_true : bool
        Parameter that controls if it should be checked whether there are active molecules and how many of them
    path_data : str, optional
        Path to data folder

    Returns
    -------
    bool
        Truth value if experiment is considered good with these settings
    """

    # load data
    data = pd.read_csv(path_data + "experiment-wise/" + str(aid) + ".csv")

    # check number of items
    if len(data) < min_number_molecules:
        return False

    # NUMBER CHECKING SECTION
    if contains_true:
        # get unique items and counts
        unique_elements, unique_counts = np.unique(
            data.activity.to_numpy(),
            return_counts=True
        )

        # check if both class labels are available here
        if unique_elements.size < 2:
            return False

        # check if number of active elements larger than provided
        if int(unique_counts[unique_elements == "active"]) < (min_number_positive_percentage * len(data)):
            return False

    # by default return true
    return True


def get_good_experiment_ids(
        number_to_sample: int = np.inf,
        min_number_molecules: int = 10000,
        contains_true: bool = True,
        min_number_positive_percentage: float = 0.01,
        path_data: str = "data/",
        random_seed: int = 131313
) -> np.ndarray:
    """
    Function that checks all experiments (already seperated into multiple csv files in folder experiment-wise) and
    checks each experiment if it has at least <param> items, has both active and inactive molecules, has at least a
    certain percentage of active molecules. Then randomly sample a few of them to reduce time for testing all
    experiments.

    Parameters
    ----------
    number_to_sample : int, optional
        Number of experiments to return (randomly), by default set to np.inf which means no sampling takes place and all
        valid experiments are returned.
    min_number_molecules : int, optional
        Minimum number of molecules an experiment has to contain to be considered good. Default: 10000
    contains_true : bool, optional
        Determines whether checks on the number of active molecules should be executed. Default: True
    min_number_positive_percentage : float, optional
        Minimum percentage of active molecules an experiment has to contain to be considered good. Default: 0.01 -> 1%
    path_data : str, optional
        Path to data folder. Default: data/
    random_seed : int, optional
        Random seed to set at beginning. Default: 131313

    Returns
    -------
    aid_list : np.ndarray
        List of experiments good and a certain number of them provided in the parameter.
    """

    # path of toc file
    path_file = path_data + "experiment-wise/ToC.npy"

    # load table of contents from file
    assert os.path.isfile(path_file)

    # load toc
    toc = np.load(path_file)

    # initialize list of good experiments
    good_aid = []

    # iterate over aids
    for aid in tqdm(toc):
        if is_experiment_good(
                aid=aid,
                min_number_molecules=min_number_molecules,
                min_number_positive_percentage=min_number_positive_percentage,
                contains_true=contains_true,
                path_data=path_data
        ):
            good_aid.append(aid)

    # if a certain number should be sampled
    if number_to_sample == np.inf:

        # return good list
        return np.array(good_aid)

    else:
        # set seed to make experiment repeatable
        np.random.seed(random_seed)

        # sample random items from list
        ret = np.random.choice(np.array(good_aid), replace=False, size=number_to_sample)
        ret.sort()
        return ret


def acquire_aid_statistical_info(
        aid: int,
        path_data: str = "data/"
) -> typing.Tuple[int, int, int]:
    """
    Function that for a provided aid, counts the total number of molecules, the active and inactive ones and returns
    the values.

    Parameters
    ----------
    aid : int
        Experiment it
    path_data : str, optional
        Path of the data where to find the data to use.

    Returns
    -------
    total : int
        Total number of molecules partaking in this experiment
    active : int
        Number of active molecules
    inactive : int
        Number of inactive molecules
    """

    # load the data from part file
    df_aid = loading.load_pure_data(aid, path_data=path_data)

    # get the activity list
    act = (df_aid.activity == 'active').to_numpy()

    # values to determine
    total = act.shape[0]
    active = int(np.sum(act))
    inactive = total - active

    # return values
    return total, active, inactive


def dataframe_add_statistical_exp_data(
        df: pd.DataFrame,
        do_in_par: bool = True,
        path_data: str = "data/"
) -> pd.DataFrame:
    """
    Function that takes a dataframe containing a aid column and fetches the experiment-related statistical information
    for each entry using the acquire_aid_statistical_info(...) function.
    Function also allows for multiprocessor usage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame - must contain an aid column or else the function will run into an error.
    do_in_par : bool, optional
        Parameter controlling if multiprocessor processing is to be used or not. By default true.
    path_data : str, optional
        Path to data folder. Default: data/

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing original data and also the statistical columns.
    """

    # copy dataframe
    df_out = df.copy()

    # get aid list
    aid_list = df_out.aid.astype(int).tolist()

    # separate if sequential or in parallel
    if do_in_par:
        with multiprocess.Pool(multiprocess.cpu_count()) as p:
            df_out[["mol_total", "mol_active", "mol_inactive"]] = p.map(acquire_aid_statistical_info, aid_list)
    else:
        df_out[["mol_total", "mol_active", "mol_inactive"]] = [
            acquire_aid_statistical_info(aid, path_data) for aid in aid_list
        ]

    return df_out


