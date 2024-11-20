import pandas as pd
import numpy as np
import random

def read_data(path : str):

    """
    Reads excel file of patients and gene expression 

    Parameters
    ----------
    path : str
        path to file

    Returns
    -------
    data : np.ndarray
        raw data, first index is the patient, second index is the gene expression

    """

    df = pd.read_excel(path)
    df_T = df.transpose() #transposing because patients are organized by column

    return df_T.to_numpy()[2:] #removing headers

def normalize_data(data: np.ndarray):

    """
    Normalizes the data by gene so the mean is 0 and the std is 1

    Parameters
    ----------
    data : np.ndarray
        Array containing the gene expression data

    Returns
    -------
    normalized_data : np.ndarray
        The normalized gene expression data

    """

    #transposing so we index the genes
    data_t = data.T
    normalized_data_t = np.zeros_like(data_t)

    for i in range(data_t.shape[0]):
        
        mean = np.mean(data_t[i])
        std = np.std(data_t[i])

        normalized_data_t[i] = (data_t[i] - mean) / std

    return normalized_data_t.T

def add_outcome(data_raw : list):

    """
    Adds outcome to the raw data array

    Parameters
    ----------
    data_raw : np.ndarray
        raw data array

    Returns
    -------
    new_data : np.ndarray
        same format as before but the first index of gene expression is now the outcome (false -> dead, true -> alive)

    """
    
    #getting size
    num_participants = data_raw.shape[0]
    num_genes = data_raw.shape[1]

    #empty array with new size
    new_data = np.zeros((num_participants, num_genes+1))

    #adding outcome of each participant
    for i in range(num_participants):
        outcome = False if i < 21 else True #21 is hard parameter from pg 24 of the supplemental info doc
        
        new_data[i] = np.append((outcome), data_raw[i])

    return new_data

def sep_data(data_raw : list, randomize : bool = False):

    """
    Seperates raw data into training and testing datasets

    Parameters
    ----------
    data_raw : np.ndarray
        raw data array with or without outcome

    randomize : bool
        randomizes seperation of data if true

    Returns
    -------
    train_data : np.ndarray
        training data, same shape as input

    test_data : np.ndarray
        testing data, same shape as input

    """

    dead_indices = list(range(0, 21))
    alive_indices = list(range(21, 60))

    #mixing up patients
    if randomize:
        random.shuffle(dead_indices)
        random.shuffle(alive_indices)

    #taking indices of first n dead/alive for training
    train_indices_dead = dead_indices[:11]
    train_indices_alive = alive_indices[:18]

    #remaining go to the test set
    test_indices_dead = dead_indices[11:]
    test_indices_alive = alive_indices[19:]

    #combine indices
    test_indices = test_indices_dead + test_indices_alive
    train_indices = train_indices_dead + train_indices_alive

    train_data = np.array([data_raw[i] for i in train_indices])
    test_data = np.array([data_raw[i] for i in test_indices])

    return train_data, test_data
    
def create_datasets(path, rand=False):

    data = read_data(path)
    data = normalize_data(data)
    data_w_outcome = add_outcome(data)

    return sep_data(data_w_outcome, randomize=rand)

if __name__ == '__main__':

    fpath = 'Instructions/Dataset_C_MD_outcome2.xlsx'

    data = read_data(fpath)

    data = add_outcome(data)

    training, testing = sep_data(data)