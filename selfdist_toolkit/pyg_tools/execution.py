import torch
import torch_geometric
import typing
from tqdm import tqdm

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

def self_distillation_procedure(
        model: torch.Module,
        self_distillation_data: typing.List[torch_geometric.data.data.Data],
        number_to_pick: int,
        device
):
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

    return None