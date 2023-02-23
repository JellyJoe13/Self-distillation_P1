import typing

import numpy as np
from sklearn.preprocessing import StandardScaler


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


def rf_scale_clean_data(
        data: np.ndarray,
        self_dest_data: typing.Union[None, np.ndarray] = None,
        col_remove_if_zero: bool = False
) -> typing.Union[
    np.ndarray,
    typing.Tuple[np.ndarray, np.ndarray]
]:
    """
    Function that removes nan and inf values, optionally cleans unnecessary data using the function clean_numpy_data()
    and then scales the data using sklearn StandardScaler.

    Parameters
    ----------
    data : np.ndarray
        Main data to be scaled.
    self_dest_data : typing.Union[None, np.ndarray], optional
        Auxiliary data such as self distillation data to be cleaned and scaled simultaneously to produce uniformly
        preprocessed data. Default None.
    col_remove_if_zero : bool, optional
        Parameter to control whether the function clean_numpy_data(...) should be applied to the data as well or not.
        Default: False.

    Returns
    -------
    data : np.ndarray
        Processed data
    self_dest_data : np.ndarray, optional
        Processed self_dest_data if supplied in parameter.
    """

    # remove nan and inf data components
    data = np.nan_to_num(data)
    self_dest_data = np.nan_to_num(self_dest_data)

    # separation to if self distillation is provided or not
    if self_dest_data is None:

        # apply function
        # if parameter is set to true, remove columns which are purely zero
        if col_remove_if_zero:
            data = clean_numpy_data(data)

        # scaling of data using sklearn standardscaler
        scaler = StandardScaler()

        # scaled data
        data = scaler.fit_transform(data)

        # return data
        return np.nan_to_num(data)

    else:
        # fuze data
        temp_fuze = np.vstack([data, self_dest_data])

        # apply function
        # if parameter is set to true, remove columns which are purely zero
        if col_remove_if_zero:
            temp_fuze = clean_numpy_data(temp_fuze)

        # scaling of data using sklearn standardscaler
        scaler = StandardScaler()

        # scaled data
        temp_fuze = scaler.fit_transform(temp_fuze)

        # de-fuze data
        data = temp_fuze[:data.shape[0]]
        self_dest_data = temp_fuze[data.shape[0]:]

        # return data and self dest data
        return np.nan_to_num(data), np.nan_to_num(self_dest_data)

