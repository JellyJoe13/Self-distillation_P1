import pandas as pd
import os


def load_pure_data(
        aid_to_load: int,
        path_data: str = "data/"
) -> pd.DataFrame:
    """
    Function that loads 'pure' data for the provided experiment (id) that was stored in a separate file in the
    preprocessing. Outputs an error if the file is not present - either because id is invalid or the preprocessing
    has not yet been run.

    Parameters
    ----------
    aid_to_load : int
        experiment id which data is to be loaded
    path_data : str, optional
        Path to data folder.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data for the experiment.
    """

    # create file path
    file_path = path_data + "experiment-wise/" + str(aid_to_load) + ".csv"

    # check if file-path exists
    if not os.path.isfile(file_path):
        raise Exception(
            'The experiment data with id {} could not be loaded. Either this experiment id is invalid or the data has '
            'yet to be split using the experiment_loadsplit(...) function.'.format(aid_to_load))

    # load and return data
    return pd.read_csv(file_path)
