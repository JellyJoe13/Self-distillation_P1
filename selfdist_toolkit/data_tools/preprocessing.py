import os
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit.Chem.Descriptors import descList
from rdkit.Chem import MolFromSmiles, RDKFingerprint
from multiprocess import Pool, cpu_count


def check_data_file(
        path_main_dataset: str,
        weburl: str = "https://ucloud.univie.ac.at/index.php/s/qRoEjX26GVH5jnx/download?path=%2F&files=df_assay_entries.csv&downloadStartSecret=0t4mnkxbz8am"
) -> None:
    """
    Function aims to check whether the main data file which is required for other opertions has already been loaded.
    If it is not loaded yet, proceed with a download from ucloud.

    Parameters
    ----------
    path_main_dataset : str
        path where the main dataset should be (put)
    weburl : str, optional
        Link from which the file could be fetched it is not yet present at the specified path. Must be a link that
        automatically triggers the download.

    Returns
    -------
    None
        Nothing
    """

    # if the file is already downloaded do nothing, else fetch it from an online source
    if os.path.isfile(path_main_dataset):
        print("Data file already present, no need for download.")
    else:
        print("File not present, begin download...")
        # begin the download of the file via the link specified in the parameter
        urlretrieve(
            weburl,
            path_main_dataset
        )
        print("Download finished.")
    return


def experiment_preprocess(
        path_main_dataset: str = "data/df_assay_entries.csv",
        path_data: str = "data/"
) -> np.ndarray:
    """
    Function that aims to preprocess the large data in a way that allows to only load specific experiment wise
    data into the program so that the program runs faster and with less memory.

    Splits data into parts related to experiments and also writes a numpy files containing all unique SMILE strings
    for later usage.

    Parameters
    ----------
    path_main_dataset : str, optional
        Path where the main dataset is stored.
    path_data : str, optional
        Path where the main data folder is placed.

    Returns
    -------
    np.ndarray
        Unique assay ids of the dataset
    """

    # load the dataset into memory
    df = pd.read_csv(path_main_dataset)

    # CHECK IF EXPERIMENT DATA FOLDER EXISTS - ELSE CREATE IT
    # but first assert if the parent folder exists
    assert os.path.exists(path_data)
    # path name to experiments folder
    path_experiments = path_data + "experiment-wise/"
    # check-create folder
    if not os.path.exists(path_experiments):
        os.makedirs(path_experiments)

    # get unique assay ids
    aid_unique = np.unique(df.aid.to_numpy())

    # save aids as a content table
    np.save(path_experiments + "ToC.npy", aid_unique)

    # iterate over aids and compute subset - save to file
    for aid in tqdm(aid_unique):
        # file name and path
        file_name = path_experiments + str(aid) + ".csv"

        # check if dataset to this has already been created
        if os.path.isfile(file_name):
            continue

        # get subset
        subset = df[df.aid == aid]
        # save subset to folder
        subset.to_csv(
            path_or_buf=file_name,
            index=False
        )

    # check if smiles mapping file is present or else write list of all smiles strings with its cid to file
    filename_smiles = "smiles.npy"
    if not os.path.isfile(path_data + filename_smiles):
        smiles_map = df[['cid', 'smiles']].sort_values(by=['cid']).drop_duplicates(subset=['cid']).to_numpy()
        np.save(
            path_data + filename_smiles,
            smiles_map
        )

    # return experiment ids
    return aid_unique


# TODO paralellize this to make it run faster
def generate_chem_smiles(
        path_main_dataset: str = "data/df_assay_entries.csv",
        path_data: str = "data/"
) -> None:
    """
    Function that if not already computed loads the dataset, identifies all unique molecules and generates chemical
    descriptor data for them using the Descriptors from rdkit from descList. The result data is written to a numpy
    file containing the data and a mapping file which maps entries from the data file to their actual molecule id(cid).

    Parameters
    ----------
    path_main_dataset : str, optional
        Path to main dataset.
    path_data : str, optional
        Path to data folder.

    Returns
    -------
    Nothing
    """

    # create saving paths
    chem_data_path = {
        "map": path_data + "chem-desc_map.npy",
        "data": path_data + "chem-desc_data.npy"
    }

    # check if data already exists
    if (not os.path.isfile(chem_data_path["map"])) and (not os.path.isfile(chem_data_path["data"])):
        print("Generating chemical descriptor data")

        # load dataframe
        df = pd.read_csv(path_main_dataset)

        # select subset of dataframe
        df = df[['cid', 'smiles']].sort_values(by=['cid']).drop_duplicates(subset=['cid']).reset_index()

        # pre-allocate storage to put data into
        storage = np.zeros((len(df), 208))

        # iterate over rows of dataset
        for idx, row in tqdm(df.iterrows()):
            storage[idx, :] = np.array([func(MolFromSmiles(row.smiles)) for _, func in descList])

        # save resulting data into files
        np.save(chem_data_path["map"], df.cid.to_numpy())
        np.save(chem_data_path["data"], storage)

    else:
        print("Chemical descriptor data already generated")

    return


def generate_chem_smiles_parallel(
        path_main_dataset: str = "data/df_assay_entries.csv",
        path_data: str = "data/"
) -> None:
    df = pd.read_csv(path_main_dataset)
    df = df[['cid', 'smiles']].sort_values(by=['cid']).drop_duplicates(subset=['cid']).reset_index()
    mols = [MolFromSmiles(x) for x in df.smiles.tolist()]
    with Pool(cpu_count()) as p:
        result = np.array([p.map(func, mols) for _, func in tqdm(descList)]).T
    return



def generate_fingerprints(
        path_main_dataset: str = "data/df_assay_entries.csv",
        path_data: str = "data/"
) -> None:
    """
    Function that if not already computed loads the dataset, identifies all unique molecules and generates the Morgan
    Fingerprint for them using the rdkit. The result data is written to a numpy file containing the data and a mapping
    file which maps entries from the data file to their actual molecule id(cid).

    Parameters
    ----------
    path_main_dataset : str, optional
        Path to main dataset.
    path_data : str, optional
        Path to data folder.

    Returns
    -------
    Nothing
    """

    # create saving paths
    fingerprint_data_path = {
        "map": path_data + "fingerprints_map.npy",
        "data": path_data + "fingerprints_data.npy"
    }

    # check if data already exists
    if (not os.path.isfile(fingerprint_data_path["map"])) and (not os.path.isfile(fingerprint_data_path["data"])):
        print("Generating fingerprints")

        # load dataframe
        df = pd.read_csv(path_main_dataset)

        # select subset of dataframe
        df = df[['cid', 'smiles']].sort_values(by=['cid']).drop_duplicates(subset=['cid']).reset_index()

        # pre-allocate storage to put data into
        storage = np.zeros((len(df), 2048))

        # iterate over rows of dataset
        for idx, row in tqdm(df.iterrows()):
            storage[idx, :] = RDKFingerprint(MolFromSmiles(row.smiles))

        # save resulting data into files
        np.save(fingerprint_data_path["map"], df.cid.to_numpy().astype(int))
        np.save(fingerprint_data_path["data"], storage.astype(bool))

    else:
        print("Fingerprints already generated")

    return


def experiment_whole_preprocess(
        path_main_dataset: str = "data/df_assay_entries.csv",
        path_data: str = "data/"
) -> np.ndarray:
    """
    Function that calls all subfunctions related to dataset loading and preprocessing in a manner that allows the
    interpreter to use at least memory as possible but possibly incurring multiple loadings of the dataset from disk.

    Parameters
    ----------
    path_main_dataset : str, optional
        Path to main dataset.
    path_data : str, optional
        Path to data folder.

    Returns
    -------
    np.ndarray
        Numpy array of unique assay ids (aid) = experiment identifiers
    """

    # check if dataset is downloaded
    check_data_file(path_main_dataset)

    # execute normal split preprocessing
    aids = experiment_preprocess(path_main_dataset, path_data)

    # generate the chemical descriptor data
    generate_chem_smiles(path_main_dataset, path_data)

    # generate the fingerprint data
    generate_fingerprints(path_main_dataset, path_data)

    return aids
