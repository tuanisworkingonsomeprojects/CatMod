import numpy as np
import pandas as pd

from .cleaning import *
from .utils import sentences_to_indices


def pipeline(X: str | list | tuple | pd.DataFrame | np.ndarray, training_predicting: str = 'training', word_to_index: dict = None, max_len: int = None, preprocess = False) -> np.ndarray:

    '''
    Description:
        This function will return the processed data which have go through the process of removing the special characters and 
        convert non-utf8 data to utf-8 data.

        Cleaning Process:
        X -> X_cleaned -> X_utf8 -> X_idx
    
    Params:
        X (str | list | tuple | pd.DataFrame | np.ndarray): The input data needed to process


    Returns:
        X (np.ndarray, dtype = bytes): The processed data
    '''

    if training_predicting == 'predicting':
        if type(X) == str:
            X = np.array([X], dtype = str)

        else:
            X = np.array(X, dtype = str)

    print('Removing Special Characters...', end = '                                                       \r')
    X_cleaned = remove_special_char(X)


    print('Removing Words that are not in the dictionary...', end = '                                       \r')
    X_cleaned = remove_word_not_in_dict(X_cleaned, word_to_index)

    
    print('Encoding utf-8...', end = '                                                                        \r')
    utf8_handler = np.vectorize(lambda x: x.encode('utf-8'))

    X_utf8 = utf8_handler(X_cleaned)

    if preprocess:
        return X_utf8
    
    print('Converting sentences to array of indices...', end = '                                            \r')
    X_idx = sentences_to_indices(X_utf8, word_to_index = word_to_index, max_len = max_len)
    

    return X_idx



    



    


