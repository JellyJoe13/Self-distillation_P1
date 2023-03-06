import typing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from selfdist_toolkit.data_tools import cleaning, sd_data_utils, loading
from selfdist_toolkit.randomforest_tools import creation


def execute_sd_rf_test_simplified(
        aid: int,
        perc_sd: float = 0.2,
        mode: str = "chem-desc",
        random_state: int = 131313
) -> typing.Dict[str, float]:
    """
    Function that is used to get the accuracy of a random forest classifier when using self distillation compared to the
    normal random forest training. Simplified because it does not use cross validation but simply splits the dataset
    into a 80:20 ratio (note that this ration is determined by the original data size and not the size including the
    self distillation based additional elements.

    Parameters
    ----------
    aid : int
        Experiment id for which to evaluate the rf and self distillation
    perc_sd : float, optional
        percentage determining how many self distillation data is to be generated and added for the student model
    mode : str, optional
        data mode, either chem-desc or fingerprint
    random_state : int, optional
        random seed to use for the random influenced functions

    Returns
    -------
    teacher_accuracy_dict : typing.Dict[str, float]
        Dictionary storing the accuracy scores of the teacher model - without self distillation
    student_accuracy_dict : typing.Dict[str, float]
        Dictionary storing the accuracy scores of the student model - with self distillation

    """

    # get random forests
    rf_teacher = creation.generate_default_rf()
    rf_student = creation.generate_default_rf()

    # LOAD DATA
    # get the prediction data
    if mode == "chem-desc":
        data, labels = loading.load_chem_desc_data(aid=aid)
    elif mode == "fingerprint":
        data, labels = loading.load_fingerprint_data(aid=aid)
    else:
        raise Exception("Mode {} is invalid. Valid options are fingerprint and chem-desc".format(mode))

    # determine number of elements to fetch for self distillation
    number_sd = int(data.shape[0] * perc_sd + 0.5)

    # get self distillation elements
    sd_data = sd_data_utils.generate_self_distillation_elements(
        aid=aid,
        number_to_generate=number_sd,
        data_gen_method=mode
    )

    # scale and clean data
    scaled_data, scaled_sdist = cleaning.rf_scale_clean_data(data, sd_data)

    # initialize accuracy storage dicts
    teacher_accuracy_dict = {}
    student_accuracy_dict = {}

    # create train and test split
    x_train, x_test, y_train, y_test = train_test_split(
        scaled_data,
        labels,
        test_size=0.2,
        random_state=random_state
    )

    # TEACHER
    # fit teacher
    rf_teacher.fit(x_train, y_train)

    # get the prediction for the test set
    pred = rf_teacher.predict(x_test)

    # get accuracy scores for teacher
    teacher_accuracy_dict["accuracy"] = accuracy_score(
        y_test,
        pred
    )
    teacher_accuracy_dict["balanced_accuracy"] = balanced_accuracy_score(
        y_test,
        pred
    )
    teacher_accuracy_dict["roc"] = roc_auc_score(
        y_test,
        pred,
        average="weighted"
    )
    teacher_accuracy_dict["precision"] = precision_score(
        y_test,
        pred,
        average="weighted"
    )
    teacher_accuracy_dict["recall"] = recall_score(
        y_test,
        pred,
        average="weighted"
    )

    # predict self distillation data
    sd_labels = rf_teacher.predict(scaled_sdist)

    # STUDENT
    # train
    rf_student.fit(
        *sd_data_utils.merge_data_self_dist_data(
            x_train,
            scaled_sdist,
            y_train,
            sd_labels
        )
    )

    # predict test items
    pred = rf_student.predict(x_test)

    # get accuracy scores for teacher
    student_accuracy_dict["accuracy"] = accuracy_score(
        y_test,
        pred
    )
    student_accuracy_dict["balanced_accuracy"] = balanced_accuracy_score(
        y_test,
        pred
    )
    student_accuracy_dict["roc"] = roc_auc_score(
        y_test,
        pred,
        average="weighted"
    )
    student_accuracy_dict["precision"] = precision_score(
        y_test,
        pred,
        average="weighted"
    )
    student_accuracy_dict["recall"] = recall_score(
        y_test,
        pred,
        average="weighted"
    )

    # return collected data
    return teacher_accuracy_dict, student_accuracy_dict


def execute_sd_rf_test(
        rf_teacher: RandomForestClassifier,
        rf_student: RandomForestClassifier,
        aid: int,
        mode: str = "chem-desc",  # alternative: fingerprint
        random_state: int = 131313,
        verbose: bool = True
) -> typing.Dict[str, typing.List[float]]:
    """
    Function that executes a random forest test for self distillation for one experiment with a data mode (fingerprint
    or chem desc data)

    Parameters
    ----------
    verbose : bool, optional
        Determines whether tqdm should be enabled or not.
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
        data_gen_method=mode
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

    # verbose control
    if verbose:
        iterating = tqdm(skf.split(scaled_data, labels))
    else:
        iterating = skf.split(scaled_data, labels)

    # for loop over splits to do cross validation
    for train_index, test_index in iterating:
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
