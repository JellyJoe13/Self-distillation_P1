from sklearn.ensemble import RandomForestClassifier


def generate_default_rf(
        random_state: int = 131313
) -> RandomForestClassifier:
    """
    Function that returns the default Random Forest to be used.

    Parameters
    ----------
    random_state : int, optional
        random state for initialization of the random forest classifier

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
    """

    # generate default RandomForest and return it
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
