import torch_geometric.data.data
import typing
import numpy as np
from selfdist_toolkit.data_tools.loading import load_pure_data
from selfdist_toolkit.pyg_tools.gnn_load import load_pyg_data_from_smiles_list
import torch
from selfdist_toolkit.pyg_tools.execution import predict


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


def self_distillation_procedure_1dim(
        model: torch.nn.Module,
        self_distillation_data: typing.List[torch_geometric.data.data.Data],
        number_to_pick: int,
        device: torch.device,
        correct_label: bool = False,
        batch_size: int = 1
) -> typing.Tuple[
    typing.List[torch_geometric.data.data.Data],
    typing.List[torch_geometric.data.data.Data]
]:
    """
    Function that takes model and self distillation candidates and predicts their labels. Most secure labels are picked
    and returned as torch_geometric.data.data.Data objects with the correct labels.

    Parameters
    ----------
    batch_size : int, optional
        batch size = how many graphs to predict at once
    model : torch.nn.Module
        Model which is to be used for predicting the labels of the data
    self_distillation_data : typing.List[torch_geometric.data.data.Data]
        List of pytorch geometric data objects to predict - candidates for self distillation items
    number_to_pick : int
        Number of most secure predictions to return
    device : torch.device
        Device on which the model should work
    correct_label : bool, optional
        Describes if a label should be put into with its value from the prediction e.g. 0.2 or if it should be corrected
        to 0. or 1.

    Returns
    -------
    self_distillation_data_list : typing.List[torch_geometric.data.data.Data]
        List of self distillation items
    remaining_candidates_list : typing.List[torch_geometric.data.data.Data]
        List of remaining self distillation candidate data objects
    """

    # get prediction for self distillation data
    sd_prediction = predict(
        model=model,
        testing_data_loader=self_distillation_data,
        device=device,
        batch_size=batch_size,
        reduce_to_hard_label=False
    )

    # pick the classes with most security = difference from 0.5
    idx_sorted = np.argsort(np.abs(sd_prediction-0.5))[::-1]

    # make list of selected and not selected data objects
    selected_data = []
    non_selected_data = [self_distillation_data[idx] for idx in idx_sorted[number_to_pick:]]

    # for the selected data set the new label
    for idx in idx_sorted[:number_to_pick]:
        # fetch data
        curr_data = self_distillation_data[idx]

        # set new label. corrected or not?
        if correct_label:
            curr_data.y = torch.tensor([1.]) if sd_prediction[idx] >= 0.5 else torch.tensor([0.])
        else:
            curr_data.y = torch.tensor([sd_prediction[idx]])

        # append data to list
        selected_data.append(curr_data)

    return selected_data, non_selected_data


def self_distillation_procedure_2dim(
        model: torch.nn.Module,
        self_distillation_data: typing.List[torch_geometric.data.data.Data],
        number_to_pick: int,
        device: torch.device,
        generation_mode: str = "actual",  # other: corrected
        batch_size: int = 1
) -> typing.Tuple[
    typing.List[torch_geometric.data.data.Data],
    typing.List[torch_geometric.data.data.Data]
]:
    """
    Function that takes model and self distillation candidates and predicts their labels. Most secure labels are picked
    and returned as torch_geometric.data.data.Data objects with the correct labels.

    Parameters
    ----------
    batch_size : int, optional
        batch size = how many graphs to predict at once
    model : torch.nn.Module
        Model which is to be used for predicting the labels of the data
    self_distillation_data : typing.List[torch_geometric.data.data.Data]
        List of pytorch geometric data objects to predict - candidates for self distillation items
    number_to_pick : int
        Number of most secure predictions to return
    device : torch.device
        Device on which the model should work
    generation_mode : str
        Mode of label generation. Actual = use actual prediction as label (direct output of model), corrected = use
        corrected output in the sense of using a one-hot encoding of the label output (label output=argmax)

    Returns
    -------
    self_distillation_data_list : typing.List[torch_geometric.data.data.Data]
        List of self distillation items
    remaining_candidates_list : typing.List[torch_geometric.data.data.Data]
        List of remaining self distillation candidate data objects
    """

    # generate prediction using prediction function
    prediction = predict(
        model=model,
        device=device,
        testing_data_loader=self_distillation_data,
        reduce_to_hard_label=False,
        batch_size=batch_size
    )

    # calculate the absolute difference between the values for each of the 2 classes
    # the larger this value the more secure the prediction is
    diff = np.abs(prediction[:, 0] - prediction[:, 1])

    # sort the differences
    diff_srt_idx = np.argsort(diff)

    # choose the <number_to_pick> elements with the largest difference (most secure elements)
    picked_elements_idx = diff_srt_idx[-number_to_pick:]

    # ==================================================================================================================
    # create self distillation elements

    # initialize return list
    self_dist_list = []

    # iterate over indexes = self distillation elements
    for idx in picked_elements_idx:
        # pick current data
        current_data = self_distillation_data[idx]

        # change label in data object
        if generation_mode == "actual":

            # input the actual prediction as a label
            current_data.y = torch.tensor(prediction[idx], dtype=torch.float)

        elif generation_mode == "corrected":

            # input the estimated class as a correct label
            current_data.y = torch.tensor([1., 0.], dtype=torch.float) if np.argmax(prediction[idx]) == 0 \
                else torch.tensor([0., 1.])

        else:
            raise Exception("Invalid generation mode parameter supplied. Either actual or corrected are possible modes")

        # append data element to list
        self_dist_list.append(current_data)

    # calculate list of elements that remain unclassified
    unclassified_list = [
        element
        for idx, element in enumerate(self_distillation_data)
        if idx not in picked_elements_idx
    ]

    # return elements and list of remaining unclassified self distillation elements
    return self_dist_list, unclassified_list
