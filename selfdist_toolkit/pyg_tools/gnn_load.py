import typing
from multiprocess import Pool, cpu_count
import torch
import numpy as np
import torch_geometric
from torch_geometric.utils.smiles import from_smiles
from selfdist_toolkit.data_tools.loading import load_pure_data


def load_all_pyg(
        path_data: str = "data/"
) -> typing.List[torch_geometric.data.data.Data]:

    # load all smiles strings
    all_smiles = np.load(path_data + "smiles.npy", allow_pickle=True)[:, 1].astype(str)

    # create dummy label for all smiles
    dummy_label = [2.] * len(all_smiles)

    # create data objects
    data_list = load_pyg_data_from_smiles_list(smiles_list=all_smiles.tolist(), label_list=dummy_label)

    # return results
    return data_list


def load_pyg_data_from_smiles(
        smiles: str,
        label: typing.Union[float, int],
        label_type: str = "hard"
) -> torch_geometric.data.data.Data:
    """
    Function that encapsulates the function form pytorch_geometric called from_smiles(...) with the functionality to
    also insert the label of the graph for graph wise prediction.

    Parameters
    ----------
    label_type : str, optional
        Label type. Either smooth (one hot encoding) or hard (single number with class value)
    smiles : str
        smiles string compressing the molecule structure
    label : typing.Union[float, int]
        label of the molecule.

    Returns
    -------
    data : torch_geometric.data.data.Data
        pytorch geometric data object containing the molecule graph and data in general
    """

    # load the data with the pyg function
    data = from_smiles(smiles)

    # todo: rethink to change it to 2-dimensional format for soft labels
    # add label information to graph
    if label_type == "hard":
        data.y = torch.tensor(np.array([label]), dtype=torch.float)
    elif label_type == "smooth":
        data.y = torch.tensor([[1., 0.]], dtype=torch.float) if label == 0. \
            else torch.tensor([[0., 1.]], dtype=torch.float)

    # return data object
    return data


def load_pyg_data_from_smiles_list(
        smiles_list: typing.List[str],
        label_list: typing.Union[typing.List[int], typing.List[float]],
        do_in_parallel: bool = True,
        label_type: str = "hard"
) -> typing.List[torch_geometric.data.data.Data]:
    """
    Function that is used to process a list of molecules and generate the pytorch geometric graph representation used
    for graph wise predictions.
    Utilizes multiprocessing to speed up generation of data.

    Parameters
    ----------
    smiles_list : typing.List[str]
        List of smiles strings
    label_list : typing.Union[typing.List[int], typing.List[float]]
        list of labels the graph data should receive
    do_in_parallel : bool, optional
        Bool controlling if multiprocessing is to be used or not.
    label_type : str, optional
        Label type. Either smooth (one hot encoding) or hard (single number with class value)

    Returns
    -------
    data_list : typing.List[torch_geometric.data.data.Data]
        list of data gnn objects.
    """

    # short list length check
    assert len(smiles_list) == len(label_list)

    # helper function to make it more accessible for iteration and for multiprocess map function
    def helper(
            input_tuple: typing.Tuple[str, typing.Union[int, float]]
    ) -> torch_geometric.data.data.Data:
        """
        Helper function for iterate processing and processing using map(...) function of multiprocessing library without
        using more complicated methods like apply or asynchronous apply.

        Parameters
        ----------
        input_tuple : typing.Tuple[str, typing.Union[int, float]]
            Tuple containing the smiles string at position 0 and label at position 1

        Returns
        -------
        data : torch_geometric.data.data.Data
            gnn data object of pytorch geometric
        """

        # calls single item function defined beforehand
        return load_pyg_data_from_smiles(*input_tuple, label_type=label_type)

    # if do in parallel - use all cores
    if do_in_parallel:

        # open multiprocessing resources
        with Pool(cpu_count()) as p:

            # map and return elements in parallel to helper function
            return p.map(helper, list(zip(smiles_list, label_list)))

    else:

        # return simple mapping for all elements
        return [
            helper(*t) for t in list(zip(smiles_list, label_list))
        ]


def load_pyg_data_aid(
        aid: int,
        do_in_parallel: bool = True,
        label_type: str = "hard"
) -> typing.List[torch_geometric.data.data.Data]:
    """
    Function that loads the pytorch geometric data objects for an experiment. Loads the smiles to load and their labels
    and then calls load_pyg_data_from_smiles_list(...) from them.
    Method to make notebooks containing such functionality look cleaner.

    Parameters
    ----------
    aid : int
        Experiment id for which to load the pyg data
    do_in_parallel : bool, optional
        Bool controlling if multiprocessing is to be used or not.
    label_type : str, optional
        Label type. Either smooth (one hot encoding) or hard (single number with class value)

    Returns
    -------
    data_list : typing.List[torch_geometric.data.data.Data]
        list of data gnn objects.
    """

    # load aid data
    aid_data = load_pure_data(aid_to_load=aid)

    # get smiles list and labels
    smiles = aid_data.smiles.tolist()
    labels = (aid_data.activity == "active").to_numpy().astype(float).tolist()

    # call function for generating the pyg elements
    pyg_data = load_pyg_data_from_smiles_list(
        smiles_list=smiles,
        label_list=labels,
        do_in_parallel=do_in_parallel,
        label_type=label_type
    )

    # return generated elements
    return pyg_data
