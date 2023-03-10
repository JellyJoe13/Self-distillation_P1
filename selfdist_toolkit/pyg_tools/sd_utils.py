import torch_geometric.data.data
import typing
import numpy as np
from selfdist_toolkit.data_tools.loading import load_pure_data
from selfdist_toolkit.pyg_tools.gnn_load import load_pyg_data_from_smiles_list


def generate_gnn_sd_data(
        aid: int,
        exclude_smiles: typing.Union[typing.List[str], np.ndarray[str]] = [],
        path_data: str = "data/",
) -> typing.List[torch_geometric.data.data.Data]:
    """
    Function that loads all molecule graphs that are not in an experiment and can thus be used for self distillation.
    Caution: returns a large number of molecules. All Data objects have a dummy label set to 2. to make a Data Loader
    work properly (signal it that this is a case of graph wise predictions).
    Function based on other function: selfdist_toolkit.data_tools.sd_data_utils.generate_self_distillation_elements(...)

    Parameters
    ----------
    aid : int
        Experiment id
    exclude_smiles : typing.Union[typing.List[str], np.ndarray[str]], optional
        List or array of smiles to exclude besides the molecule stored in the experiment
    path_data : str, optional
        Path to data folder

    Returns
    -------
    data_list : typing.List[torch_geometric.data.data.Data]
        List of gnn data objects representing molecules
    """
    # function closely related to function

    # auto-convert input if necessary
    if type(exclude_smiles) == list:
        exclude_smiles = np.array(exclude_smiles, dtype=str)

    # load original dataset
    data = load_pure_data(aid_to_load=aid, path_data=path_data)

    # extract smiles strings
    existing_smiles = data.smiles.to_numpy().astype(str)

    # concatenate exclude and existing_smiles
    existing_smiles = np.concatenate([existing_smiles, exclude_smiles], dtype=str)

    # load list of all smiles
    all_smiles = np.load(path_data + "smiles.npy", allow_pickle=True)[:, 1].astype(str)

    # get smiles which are not tested for the experiment
    free_smiles = all_smiles[np.isin(all_smiles, existing_smiles, invert=True)]

    # generate pyg data with dummy label (because without dummy label the dataloader will not work properly)
    dummy_label = [2.] * len(free_smiles)
    data_list = load_pyg_data_from_smiles_list(smiles_list=list(free_smiles), label_list=dummy_label)

    # return results
    return data_list
