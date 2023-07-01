"""
The content in this file is inspired by content shared in the github repository
https://github.com/Bjarten/early-stopping-pytorch.
"""
import numpy as np
from selfdist_toolkit.pyg_tools.accuracy import AccuracyStorage


class EarlyStopBasic:
    """
    Class to store metrics for when to stop and implement logic when to actually stop the training.
    """
    def __init__(
            self,
            min_improvement: float,
            num_epoch_wait: int,
            measurement: str
    ):
        self.min_improvement = min_improvement
        self.num_epoch_wait = num_epoch_wait
        self.measurement = measurement
        self.counter = 0
        self.best_metric_val = 0

    def new_epoch_test_stop(
            self,
            accuracy_storage: AccuracyStorage
    ) -> bool:
        """
        Supply current metric scores and determine whether to stop or not. Sees if it is a new maximum score:
         - if it is then set the counter to 0
         - if not then increase the counter by 1 and if this is larger than the set parameter then indicate to stop.

        Parameters
        ----------
        accuracy_storage : AccuracyStorage
            Class storing the recoded metrics of the last epoch

        Returns
        -------
        bool
            True if the training should be stopped
        """

        # check if selected measure is in the accuracy storage
        if self.measurement not in accuracy_storage.to_dict():
            raise Exception("selected measurement not in accuracy storage element. Check selection or accuracy "
                            "storage element")

        # get the metric value
        metric_val = accuracy_storage.to_dict()[self.measurement]

        # check if element is better than observed maximum (plus delta) or not
        if metric_val > (self.best_metric_val + self.min_improvement):

            # it is better - set the new best and reset counter
            self.best_metric_val = metric_val
            self.counter = 0

            # return false, obviously we do not want to stop
            return False

        else:

            # it is not better, increase counter
            self.counter += 1

            # if this happened n times (parameter controlled) then tell the calling program to stop
            if self.counter >= self.num_epoch_wait:
                return True

