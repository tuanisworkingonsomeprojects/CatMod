import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def read_glove_vec(file_path: str, to_list: bool = False) -> tuple:
    '''
    params:
        file_path (str): the path of the GloVe word Embedding Matrix
        to_list (bool): if it is set to True the return value for words will be 'list' instead of 'set'
        
    returns:
        words (set or list): a list of words in the Embedding Matrix
        word_to_vec_map (dict): a dictionary of Embedding vectors according to the given words
    '''
    
    words = set()
    word_to_vec_map = dict()
    
    with open(file_path, 'rb') as f:
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    if to_list:
        words = sorted(list(words))
    return words, word_to_vec_map


def get_dataset_from_csv(file_path: str, X_column_name: str, Y_column_name: str, dropna: bool = True) -> pd.DataFrame: 
    '''
    Description:
        The function will load the csv file in and return the corresponding pd.DataFrame
    
    params:
        file_path (str): The path of the csv file
        X_column_name (str): The name of the X column
        Y_column_name (str): The name of the Y column
        dropna (bool): Default True
            True: Drop Nan or Null Values
            False: Keep Nan or Null Values
        
    returns:
        dataset (pd.DataFrame): The dataset without Nan or Null values
    '''
    
    dataset = pd.read_csv(file_path, index_col = False)
    dataset = dataset[[X_column_name, Y_column_name]]
    
    if dropna:
        dataset.dropna(inplace = True)
    
    dataset.rename(columns = {X_column_name: 'X', Y_column_name: 'Y'}, inplace = True)
    dataset.index = range(len(dataset))
    return dataset



def dataset_to_XY(df: pd.DataFrame) -> np.array:
    '''
    params:
        df (pd.DataFrame): is a processed dataset with 2 column 'X' and 'Y' labeled accordingly

    returns:
        X, Y (np.array): 2 numpy arrays
    '''


    X = np.asarray(df['X'].apply(str.lower), dtype = '<U52')
    Y = np.asarray(df['Y'].apply(str.lower), dtype = '<U52')

    return X, Y


def get_train_test_dataset(df: pd.DataFrame, test_size: float = 0.2, random_seed: int = 42, to_XY: bool = False) -> pd.DataFrame:
    '''
    params:

    returns:

    
    '''
    
    train_data, test_data = train_test_split(df, test_size = test_size, random_state = random_seed)
    if to_XY:
        X_train, Y_train =  dataset_to_XY(train_data)
        X_test, Y_test = dataset_to_XY(test_data)
        return X_train, Y_train, X_test, Y_test

    return train_data, test_data