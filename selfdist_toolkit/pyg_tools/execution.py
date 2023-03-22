import torch
import torch_geometric
import typing
from tqdm import tqdm
import numpy as np


# code inspired by https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/main_pyg.py and

def training(
        model: torch.nn.Module,
        trainings_data: typing.Union[
            torch_geometric.loader.DataLoader,
            typing.List[torch_geometric.data.data.Data]
        ],
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_criterion,
        batch_size: int = 1
) -> np.ndarray[float]:
    """
    Function used for training the model on the supplied training data.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    trainings_data : typing.Union[torch_geometric.loader.DataLoader,typing.List[torch_geometric.data.data.Data]]
        Data to train the model with
    device : torch.device
        Device on which the model should operate
    optimizer : torch.optim.optimizer.Optimizer
        Optimizer used for training
    loss_criterion
        Loss generating class/function
    batch_size : int, optional
        Batch size in case a list of pytorch geometric objects is supplied. Default: 1

    Returns
    -------
    mean_loss : np.ndarray[float]
        Mean over the batch wise losses
    """

    # set model to train mode
    model.train()

    # save losses for statistical purposes
    loss_storage = []

    # if not a dataloader is provided create it
    if not type(trainings_data) == torch_geometric.loader.DataLoader:
        trainings_data = torch_geometric.loader.DataLoader(trainings_data, batch_size=batch_size)

    # iterate over batches
    for batch in tqdm(trainings_data):

        # clear gradient
        optimizer.zero_grad()

        # transfer data to device
        batch = batch.to(device)

        # calculate the labels
        y_pred = model(batch).flatten()

        # use the loss criterion to generate the loss
        loss = loss_criterion(y_pred, batch.y)

        # train
        loss.backward()

        # optimization step
        optimizer.step()

        # append loss to saving
        loss_storage.append(loss.detach().cpu().numpy())

    return np.mean(np.array(loss_storage))


def predict(
        model: torch.nn.Module,
        testing_data_loader: typing.Union[
            torch_geometric.loader.DataLoader,
            typing.List[torch_geometric.data.data.Data]
        ],
        device,
        reduce_to_hard_label: bool = False,
        batch_size: int = 1  # only if no loader is supplied
) -> np.ndarray[float]:
    """
    Function that given the inputs computes the label (soft or hard) of the inputted data using the inputted model.

    Parameters
    ----------
    model : torch.nn.Module
        Model used for prediction - will not be trained
    testing_data_loader : typing.Union[torch_geometric.loader.DataLoader,typing.List[torch_geometric.data.data.Data]]
        test data in the form of a list of pytorch geometric data objects or a data loader
    device : torch.device
        device on which the neural network should act on
    reduce_to_hard_label : bool, optional
        Controls if the output should be in the form of a single label or the actual 2-feature output containing a value
        /probability/certainty of the element being a specific class. Default: False - no reduce
    batch_size : int, optional
        Batch size used in the Data Loader in case a List of data objects is provided instead of a DataLoader

    Returns
    -------
    prediction : np.ndarray[float]
        Prediction with dimension either: num_graphs x 2 or num_graphs
    """

    # question to myself: use hard label or soft label for measure? like active/inactive or active:0.3, inactive: 0.9?

    # if input data_loader is not yet a loader - transform it into one
    if not type(testing_data_loader) == torch_geometric.loader.DataLoader:
        testing_data_loader = torch_geometric.loader.DataLoader(testing_data_loader, batch_size=batch_size)

    # set the model to evaluation mode
    model.eval()

    # setup prediction list
    prediction = []

    # iterate over batches
    for batch in tqdm(testing_data_loader):
        # transfer batch to device
        batch = batch.to(device)

        # get prediction without loss generation
        with torch.no_grad():
            pred = torch.nn.Sigmoid()(model(batch).flatten())

        # transfer result back to cpu and append it to list
        prediction.append(pred.detach().cpu())

    # fuze data
    prediction = torch.cat(prediction, dim=0).numpy()

    # if output reduction is specified transform it to unidimensional prediction
    if reduce_to_hard_label:
        prediction = (prediction >= 0.5).astype(int)

    # return prediction
    return prediction
