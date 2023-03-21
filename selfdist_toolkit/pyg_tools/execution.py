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
            curr_data.y = sd_prediction[idx]

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
