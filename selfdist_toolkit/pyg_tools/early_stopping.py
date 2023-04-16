"""
The content in this file is inspired by content shared in the github repository
https://github.com/Bjarten/early-stopping-pytorch.
"""
import numpy as np
from selfdist_toolkit.pyg_tools.accuracy import AccuracyStorage


class EarlyStopBasic:
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

