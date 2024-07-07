import numpy as np




def np_array_converter(X):
    utf8_handler = np.vectorize(lambda x: x.encode('utf-8'))

    if type(X) == str:
        X = np.array([X], dtype = str)
        X = utf8_handler(X)

    else:
        X = np.array(X, dtype = str)
        X = utf8_handler(X)

    return X