import typing

import pandas as pd
import os
import numpy as np
from selfdest_toolkit.data_tools.cleaning import clean_numpy_data


def load_pure_data(
        aid_to_load: int,
        path_data: str = "data/"
) -> pd.DataFrame:
    """
    Function that loads 'pure' data for the provided experiment (id) that was stored in a separate file in the
    preprocessing. Outputs an error if the file is not present - either because id is invalid or the preprocessing
    has not yet been run.

    Parameters
    ----------
    aid_to_load : int
        experiment id which data is to be loaded
    path_data : str, optional
        Path to data folder.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data for the experiment.
    """

    # create file path
    file_path = path_data + "experiment-wise/" + str(aid_to_load) + ".csv"

    # check if file-path exists
    if not os.path.isfile(file_path):
        raise Exception(
            'The experiment data with id {} could not be loaded. Either this experiment id is invalid or the data has '
            'yet to be split using the experiment_loadsplit(...) function.'.format(aid_to_load))

    # load and return data
    return pd.read_csv(file_path)


def load_chem_desc_data(
        aid: int,
        path_data: str = "data/"
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Function that loads the chemical descriptor dataset for the provided experiment (id). Also cleans the data of
    repetitive or unnecessary data.
    Addition: due to timing reasons the precomputed results will be saved and loaded if possible.

    Parameters
    ----------
    aid : int
        Experiment id for which the data is to be fetched
    path_data : str, optional
        Path to data folder

    Returns
    -------
    np.ndarray
        chemical descriptor data
    np.ndarray
        labels of the data (0 for inactive, 1 for active)
    """

    # PRESAVING AND FETCHING PART
    # path where the data would be prestored
    path_chemdata = path_data + "precomputed/chemdata/"

    # check if folder present
    if not os.path.exists(path_chemdata):
        os.makedirs(path_chemdata)

    # check if file already exists and can be loaded instead of creating it
    file_names = {
        "data": str(aid) + "_data.npy",
        "label": str(aid) + "_label.npy"
    }
    if os.path.isfile(path_chemdata + file_names["data"]) and os.path.isfile(path_chemdata + file_names["label"]):
        return np.load(path_chemdata + file_names["data"]).astype(float), \
               np.load(path_chemdata + file_names["label"]).astype(int)

    # NORMAL LOADING PART
    # load pure data
    loaded_data = load_pure_data(
        aid_to_load=aid,
        path_data=path_data
    )

    # load pure chemical descriptor data
    chem_data_map = np.load(path_data + "chem-desc_map.npy")
    chem_data_data = np.load(path_data + "chem-desc_data.npy")

    # cleanup of data - data may contain rows with always the same value, only 0s, etc.
    chem_data_data = clean_numpy_data(chem_data_data)

    # map the cids to numpy array to get the temporary data
    data = np.stack(
        loaded_data.cid.map(
            lambda x:
                chem_data_data[chem_data_map == x][0]
        ).to_numpy()
    )

    # fetch the labels of the data elements:
    labels = loaded_data.activity.map(lambda x: int(x == "active")).to_numpy()

    # save the generated data to disk
    np.save(path_chemdata + file_names["data"], data)
    np.save(path_chemdata + file_names["label"], labels.astype(int))

    # return data and labels
    return data, labels


def load_fingerprint_data(
        aid: int,
        path_data: str = "data/"
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Function that loads the Morgan Fingerprint dataset for the provided experiment (id). Also cleans the data of
    repetitive or unnecessary data.
    Addition: due to timing reasons the precomputed results will be saved and loaded if possible.

    Parameters
    ----------
    aid : int
        Experiment id for which the data is to be fetched
    path_data : str, optional
        Path to data folder

    Returns
    -------
    np.ndarray
        chemical descriptor data
    np.ndarray
        labels of the data (0 for inactive, 1 for active)
    """

    # PRESAVING AND FETCHING PART
    # path where the data would be prestored
    path_fingerprint = path_data + "precomputed/fingerprint/"

    # check if folder present
    if not os.path.exists(path_fingerprint):
        os.makedirs(path_fingerprint)

    # check if file already exists and can be loaded instead of creating it
    file_names = {
        "data": str(aid) + "_data.npy",
        "label": str(aid) + "_label.npy"
    }
    if os.path.isfile(path_fingerprint + file_names["data"]) and os.path.isfile(path_fingerprint + file_names["label"]):
        return np.load(path_fingerprint + file_names["data"]).astype(float), \
               np.load(path_fingerprint + file_names["label"]).astype(int)

    # NORMAL LOADING PART
    # load the pure data
    loaded_data = load_pure_data(
        aid_to_load=aid,
        path_data=path_data
    )

    # load pure chemical descriptor data
    fingerprint_map = np.load(path_data + "fingerprints_map.npy")
    fingerprint_data = np.load(path_data + "fingerprints_data.npy")

    # cleanup of data - data may contain rows with always the same value, only 0s, etc.
    fingerprint_data = clean_numpy_data(fingerprint_data)

    # map the cids to numpy array to get the temporary data
    data = np.stack(
        loaded_data.cid.map(
            lambda x:
            fingerprint_data[fingerprint_map == x][0]
        ).to_numpy()
    )

    # fetch the labels of the data elements:
    labels = loaded_data.activity.map(lambda x: int(x == "active")).to_numpy()

    # save the generated data to disk
    np.save(path_fingerprint + file_names["data"], data.astype(bool))
    np.save(path_fingerprint + file_names["label"], labels.astype(int))

    # return data and labels
    return data, labels


def preload_chem_data_all(
        aid_list: np.ndarray,
        path_data: str = "data/"
) -> None:
    """
    Function preloads all the chemical descriptor data and makes sure it is written to disk beforehand to speed up
    further computation.

    Parameters
    ----------
    aid_list : np.ndarray
        list of experiment ids to preload
    path_data : str, optional
        path of data folder

    Returns
    -------
    Nothing
    """

    # iterate over list of aids
    for aid in aid_list:
        load_chem_desc_data(aid, path_data)

    return


def preload_fingerprint_data_all(
        aid_list: np.ndarray,
        path_data: str = "data/"
) -> None:
    """
    Function preloads all the fingerprint data and makes sure it is written to disk beforehand to speed up further
    computation.

    Parameters
    ----------
    aid_list : np.ndarray
        list of experiment ids to preload
    path_data : str, optional
        path of data folder

    Returns
    -------
    Nothing
    """

    # iterate over list of aids
    for aid in aid_list:
        load_chem_desc_data(aid, path_data)

    return
