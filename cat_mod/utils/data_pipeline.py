import numpy as np
import pandas as pd

from .cleaning import *


def pipeline(X: str | list | tuple | pd.DataFrame, training_predicting: str = 'training'):

    if training_predicting == 'predicting':
        if type(X) == str:
            X = np.array([X], dtype = str)

        else:
            X = np.array(X, dtype = str)

    X = remove_special_char(X)

    utf8_handler = np.vectorize(lambda x: x.encode('utf-8'))

    X = utf8_handler(X)

    return X



    



    


