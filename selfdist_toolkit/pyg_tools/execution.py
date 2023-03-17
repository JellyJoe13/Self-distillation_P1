import torch
import torch_geometric
import typing
from tqdm import tqdm
import numpy as np


# code inspired by https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/main_pyg.py and
#

def training(
        GNN: torch.Module,
        trainings_data
):
    return None


def validating(
        GNN: torch.Moduel,
        validation_data
):
    return None


def testing(
        GNN: torch.Module,
        testing_data
):
    return None


def self_distillation_procedure_soft(
        model: torch.Module,
        self_distillation_data: typing.List[torch_geometric.data.data.Data],
        number_to_pick: int,
        device,
        generation_mode: "actual"  # other: corrected
) -> typing.Tuple[
    typing.List[torch_geometric.data.data.Data],
    typing.List[torch_geometric.data.data.Data]
]:
    # set the model to evaluation mode
    model.eval()

    # initialize lists containing the elements predictions
    prediction = []

    # iterate over batches
    for batch in tqdm(torch_geometric.loader.DataLoader(self_distillation_data)):
        # transfer batch to device
        batch = batch.to(device)

        # get prediction without loss generation
        with torch.no_grad():
            pred = model(batch)

        # transfer result back to cpu and append it to list
        prediction.append(pred.detach().cpu())

    # fuze data
    prediction = torch.cat(prediction, dim=0).numpy()

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
    mask = np.ones(self_distillation_data.shape, dtype=bool)
    mask[picked_elements_idx] = False
    unclassified_list = self_distillation_data[mask]

    # return elements and list of remaining unclassified self distillation elements
    return self_dist_list, unclassified_list
