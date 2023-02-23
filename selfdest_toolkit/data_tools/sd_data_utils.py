import typing

import numpy as np
from selfdest_toolkit.data_tools.loading import load_pure_data


def generate_self_distillation_elements(
        aid: int,
        number_to_generate: int,
        path_data: str = "data/",
        seed: int = 131313,
        data_gen_method: str = "chem-desc"  # other: fingerprint
) -> np.ndarray:
    """
    Takes some random molecules which are not yet in the experiment scope (with the id provided) and load fingerprint or
    chemical descriptor data, then return it.
    Note: using other molecules as else it would need be removed from the original dataset (which may already be very
    small and hence it would not be beneficial to remove further elements)

    Parameters
    ----------
    aid : int
        Experiment id for which self distillation elements are to be generated
    number_to_generate : int
        number of elements to fetch for self distillation
    path_data : str, optional
        Path to data folder
    seed : int
        Seed to set; by default set to 131313, if None, then no seed will be set. Seed recommended for reproducability.
    data_gen_method : str
        Defines which data mode is active: either 'fingerprint' or 'chem-desc' for corresponding data type.

    Returns
    -------
    molecule_data : np.ndarray
        Fetched self distillation data
    """

    # set the seed if parameter is not None
    if seed is not None:
        np.random.seed(seed)

    # load the original data to see which molecules are already used
    # Note: using other molecules as else it would need to be removed from the original dataset (which may already not

    # have that much elements to begin with
    data = load_pure_data(aid_to_load=aid, path_data=path_data)

    # get the cids from the dataset out of it
    cids = data.cid.to_numpy()

    # load list of all cids
    cid_list = np.load(path_data + "smiles.npy", allow_pickle=True)[:, 0].astype(int)

    # change it into list of cids not contained yet in the experiment
    cid_list = cid_list[np.isin(cid_list, cids, invert=True)]

    # select some elements to final data generation
    selected = np.random.choice(cid_list, size=number_to_generate, replace=False)
    selected.sort()

    # DATA LOADING
    # load the data
    if data_gen_method == "fingerprint":
        data = np.load(path_data + "fingerprints_data.npy").astype(int)
        dmap = np.load(path_data + "fingerprints_map.npy").astype(int)
    elif data_gen_method == "chem-desc":
        data = np.load(path_data + "chem-desc_data.npy")
        dmap = np.load(path_data + "chem-desc_map.npy").astype(int)
    else:
        raise Exception("No valid option was chosen for the parameter data_gen_method.")

    # get the data corresponding to the chosen molecules
    # get position in whole list
    pos = np.searchsorted(dmap, selected)

    # collect data
    molecule_data = data[pos]

    return molecule_data


def merge_data_self_dist_data(
        data: np.ndarray,
        sd_data: np.ndarray,
        label: np.ndarray,
        sd_label: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:

    # fuze data
    new_data = np.vstack([data, sd_data])

    # fuze labels
    new_labels = np.concatenate([label, sd_label])

    return new_data, new_labels
