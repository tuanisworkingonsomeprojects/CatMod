import re
import numpy as np


def remove_special_char(X: np.ndarray):
    '''
    Description:

    Params:
        X (np.ndarray): shape (number of examples, ). dtype = str

    Returns:


    '''

    SPECIAL_CHARACTER = 'r[^\w\s]'
    SPECIAL_CHARACTER = re.compile(SPECIAL_CHARACTER, re.X)
    remover = np.vectorize(lambda x: re.sub(SPECIAL_CHARACTER, "", x))
    X = remover(X)
    return X


    






