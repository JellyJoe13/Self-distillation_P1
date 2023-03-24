from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
import numpy as np
import pandas as pd
import typing


class AccuracyStorage:
    def __init__(
            self,
            accuracy: float,
            balanced_accuracy: float,
            roc_score: float,
            precision: float,
            recall: float
    ):
        # assigning values
        self.accuracy_score = accuracy
        self.balanced_accuracy = balanced_accuracy
        self.roc_score = roc_score
        self.precision = precision
        self.recall = recall

    def to_dict(self) -> typing.Dict[str, float]:
        return {
            "accuracy_score": self.accuracy_score,
            "balanced_accuracy": self.balanced_accuracy,
            "roc_score": self.roc_score,
            "precision": self.precision,
            "recall": self.recall
        }

    def to_df(self, index):
        return pd.DataFrame(self.to_dict(), index=[index])


def calculate_accuracies_1d(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> AccuracyStorage:

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
        average="weighted"
    )
    recall = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average="weighted"
    )

    # generate accuracy storage class
    return AccuracyStorage(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        roc_score=roc,
        precision=precision,
        recall=recall
    )
