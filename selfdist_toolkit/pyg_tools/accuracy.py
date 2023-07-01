import torch
import torch_geometric.loader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import matthews_corrcoef
import numpy as np
import pandas as pd
import typing


class AccuracyStorage:
    """
    Accuracy holding class, works similarly to dictionary. Can be used to retrieve pandas dataframe or dict.
    """
    def __init__(
            self,
            accuracy: float,
            balanced_accuracy: float,
            roc_score: float,
            precision: float,
            recall: float,
            mcc: float = 0
    ):
        # assigning values
        self.accuracy_score = accuracy
        self.balanced_accuracy = balanced_accuracy
        self.roc_score = roc_score
        self.precision = precision
        self.recall = recall
        self.mcc = mcc

    def to_dict(self) -> typing.Dict[str, float]:
        """
        Returns stored values as dict.

        Returns
        -------
        typing.Dict[str, float]
            Dict containing stored values
        """
        return {
            "accuracy_score": self.accuracy_score,
            "balanced_accuracy": self.balanced_accuracy,
            "roc_score": self.roc_score,
            "precision": self.precision,
            "recall": self.recall,
            "mcc": self.mcc
        }

    def to_df(self, index):
        """
        Returns stored values as pandas dataframe with row index set to parameter input.

        Parameters
        ----------
        index : int
            Future row index of dataframe

        Returns
        -------
        pd.DataFrame
            Dataframe containing stored values
        """
        return pd.DataFrame(self.to_dict(), index=[index])


def helper_pyg_to_numpy_label(
        data_loader: torch_geometric.loader.DataLoader,
        correct_label: bool = False
) -> np.ndarray[int]:
    """
    Helper function to extract labels from DataLoader object, currently always corrects labels to hard labels because
    mainly used for true label fetching which is for some accuracy measurements important to be integer.

    Parameters
    ----------
    data_loader : torch_geometric.loader.DataLoader
        Data loader from which to extract labels
    correct_label : bool, optional
        Determine whether labels are to be corrected using 0.5 decision threshold or not. Default False

    Returns
    -------
    np.ndarray[int]
        List of labels
    """

    # decide if the labels need to be converted to hard labels
    if correct_label:
        return torch.concat([
            torch.tensor([1.], dtype=torch.float) if batch.y >= 0.5 else torch.tensor([0.], dtype=torch.float)
            for batch in data_loader
        ]).detach().cpu().numpy().astype(int)
    else:
        # iterate over dataloader and concatenate labels
        return torch.concat([batch.y for batch in data_loader]).detach().cpu().numpy().astype(int)


def calculate_accuracies_1d(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> AccuracyStorage:
    """
    Calculates the following accuracy metrics and returns them in an AccuracyStorage class object: accuracy, balanced
    accuracy, roc_auc, precision, recall, mcc.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        predicted labels

    Returns
    -------
    AccuracyStorage
        Element holding metric scores similarly to dictionary.
    """

    # fetch the accuracy scores
    accuracy = accuracy_score(
        y_true=y_true,
        y_pred=y_pred
    )
    balanced_accuracy = balanced_accuracy_score(
        y_true=y_true,
        y_pred=y_pred
    )
    roc = roc_auc_score(
        y_true=y_true,
        y_score=y_pred,
        average="weighted"
    )
    precision = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average="weighted"
    )
    mcc = matthews_corrcoef(
        y_true=y_true,
        y_pred=y_pred
    )

    # generate accuracy storage class
    return AccuracyStorage(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        roc_score=roc,
        precision=precision,
        recall=recall,
        mcc=mcc
    )
