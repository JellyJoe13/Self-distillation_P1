import numpy as np


def clean_numpy_data(
        data: np.ndarray
) -> np.ndarray:
    """
    Function that cleans data by doing some operations. Currently implemented: remove zero columns (features).

    Parameters
    ----------
    data : np.ndarray
        Data which is to be cleaned column wise.

    Returns
    -------
    data : np.ndarray
        cleaned data
    """

    # remove zero columns
    data = data[:, np.invert((data == 0).all(axis=0))]

    # return last stage of data
    return data
