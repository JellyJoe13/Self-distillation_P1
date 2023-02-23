import os.path
import numpy as np
import pandas as pd
from tqdm import tqdm


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
