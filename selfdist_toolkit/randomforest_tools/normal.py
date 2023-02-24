from sklearn.ensemble import RandomForestClassifier
import typing
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from selfdist_toolkit.data_tools import loading, cleaning


def execute_normal_rf_test(
        rf: RandomForestClassifier,
        aid: int,
        mode: str = "chem-desc",  # alternative: fingerprint
        random_state: int = 131313
) -> typing.Dict[str, typing.List[float]]:
    """
    Function that executes a random forest test for non self distillation for one experiment with a data mode
    (fingerprint or chem desc data)

    Parameters
    ----------
    rf : RandomForestClassifier
        Random Forest to use for the testing process
    aid : int
        Experiment to use for the testing process
    mode : str, optional
        Data Mode. Either fingerprint or chem-desc
    random_state : int, optional
        Random state to use for random-influenced functions

    Returns
    -------
    accuracy_storage : typing.Dict[str, typing.List[float]]
        Dict in which the accuracy scores are to be collected. Stores accuracy, balanced accuracy, roc, precision,
        recall
    """

    # load data for experiment
    if mode == "chem-desc":
        data, labels = loading.load_chem_desc_data(aid)
    elif mode == "fingerprint":
        data, labels = loading.load_fingerprint_data(aid)
    else:
        raise Exception("Mode {} is invalid. Valid options are fingerprint and chem-desc".format(mode))

    # scale and clean data
    scaled_data = cleaning.rf_scale_clean_data(data)

    # initialize storage dict
    accuracy_storage = {
        "accuracy": [],
        "balanced_accuracy": [],
        "roc": [],
        "precision": [],
        "recall": []
    }

    # generate Split
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    # for loop over splits to do cross validation
    for train_index, test_index in tqdm(skf.split(scaled_data, labels)):

        # split dataset:
        x_train, x_test = scaled_data[train_index], scaled_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # random forest fit data
        rf.fit(x_train, y_train)

        # generate prediction
        pred = rf.predict(x_test)

        # calculate scores and append them
        accuracy_storage["accuracy"].append(
            accuracy_score(y_test, pred)
        )
        accuracy_storage["balanced_accuracy"].append(
            balanced_accuracy_score(y_test, pred)
        )
        accuracy_storage["roc"].append(
            roc_auc_score(y_test, pred, average="weighted")
        )
        accuracy_storage["precision"].append(
            precision_score(y_test, pred, average="weighted")
        )
        accuracy_storage["recall"].append(
            recall_score(y_test, pred, average="weighted")
        )

    return accuracy_storage
