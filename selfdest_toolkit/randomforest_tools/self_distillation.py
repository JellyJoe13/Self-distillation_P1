import typing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from selfdest_toolkit.data_tools import cleaning, sd_data_utils, loading


def execute_normal_rf_test(
        rf_teacher: RandomForestClassifier,
        rf_student: RandomForestClassifier,
        aid: int,
        mode: str = "chem-desc",  # alternative: fingerprint
        random_state: int = 131313
) -> typing.Dict[str, typing.List[float]]:
    """
    Function that executes a random forest test for self distillation for one experiment with a data mode (fingerprint
    or chem desc data)

    Parameters
    ----------
    rf_teacher : RandomForestClassifier
        Teacher Random Forest
    rf_student : RandomForestClassifier
        Student Random Forest
    aid : int
        Experiment to use for the testing process
    mode : str, optional
        Data Mode. Either fingerprint or chem-desc
    random_state : int, optional
        Random state to use for random-influenced functions

    Returns
    -------
    normal_accuracy_storage : typing.Dict[str, typing.List[float]]
        Dict in which the accuracy scores of the non self distillation approach are to be collected. Stores accuracy,
        balanced accuracy, roc, precision, recall
    dist_accuracy_storage : typing.Dict[str, typing.List[float]]
        Dict in which the accuracy scores of the self distillation approach are to be collected. Stores accuracy,
        balanced accuracy, roc, precision, recall
    """

    # LOAD DATA
    # get the prediction data
    if mode == "chem-desc":
        data, labels = loading.load_chem_desc_data(aid=aid)
    elif mode == "fingerprint":
        data, labels = loading.load_fingerprint_data(aid=aid)
    else:
        raise Exception("Mode {} is invalid. Valid options are fingerprint and chem-desc".format(mode))

    # determine number of elements to fetch for self distillation
    number_sd = int(data.shape[0] * 0.2 + 0.5)

    # get self distillation elements
    sd_data = sd_data_utils.generate_self_distillation_elements(
        aid=aid,
        number_to_generate=number_sd,
        data_gen_method="chem-desc"
    )

    # scale and clean data
    scaled_data, scaled_sdist = cleaning.rf_scale_clean_data(data, sd_data)

    # initialize normal and self distillation storage dict
    normal_accuracy_storage = {
        "accuracy": [],
        "balanced_accuracy": [],
        "roc": [],
        "precision": [],
        "recall": []
    }
    dist_accuracy_storage = {
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

        # TEACHER RANDOMFOREST
        # training
        rf_teacher.fit(x_train, y_train)

        # predicting self distillation elements
        selfdist_label = rf_teacher.predict(scaled_sdist)

        # predicting test
        pred = rf_teacher.predict(x_test)

        # calculate scores and append them
        normal_accuracy_storage["accuracy"].append(
            accuracy_score(y_test, pred)
        )
        normal_accuracy_storage["balanced_accuracy"].append(
            balanced_accuracy_score(y_test, pred)
        )
        normal_accuracy_storage["roc"].append(
            roc_auc_score(y_test, pred, average="weighted")
        )
        normal_accuracy_storage["precision"].append(
            precision_score(y_test, pred, average="weighted")
        )
        normal_accuracy_storage["recall"].append(
            recall_score(y_test, pred, average="weighted")
        )

        # STUDENT RANDOMFOREST
        # training
        rf_student.fit(*sd_data_utils.merge_data_self_dist_data(x_train, scaled_sdist, y_train, selfdist_label))

        # predicting test
        pred = rf_student.predict(x_test)

        # calculate scores and append them
        dist_accuracy_storage["accuracy"].append(
            accuracy_score(y_test, pred)
        )
        dist_accuracy_storage["balanced_accuracy"].append(
            balanced_accuracy_score(y_test, pred)
        )
        dist_accuracy_storage["roc"].append(
            roc_auc_score(y_test, pred, average="weighted")
        )
        dist_accuracy_storage["precision"].append(
            precision_score(y_test, pred, average="weighted")
        )
        dist_accuracy_storage["recall"].append(
            recall_score(y_test, pred, average="weighted")
        )

    return normal_accuracy_storage, dist_accuracy_storage
