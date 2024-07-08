import re
import numpy as np


def remove_special_char(X: np.ndarray):
    '''
    Description:

    Params:
        X (np.ndarray): shape (number of examples, ). dtype = str

    Returns:
        X (np.ndarray): cleaned string without special character

    '''

    SPECIAL_CHARACTER = 'r[^\w\s]'
    SPECIAL_CHARACTER = re.compile(SPECIAL_CHARACTER, re.X)
    remover = np.vectorize(lambda x: re.sub(SPECIAL_CHARACTER, '', x))
    X = remover(X)

    return X

def remove_word_not_in_dict(X: np.ndarray, word_to_index: dict):
    '''
    Description:

    Params:
        X (np.ndarray): shape (number of examples, ). dtype = str

    Returns:
        X (np.ndarray): array of string that remove words not in the word_to_index dictionary

    '''
    X = np.array(X, dtype = str)

    

    def remover_func(x, word_to_index):
        
        x = str(x)
        x = x.strip()
        # Encode the whole sentence to bytes
        x = x.encode('utf-8') # Type bytes


        words = word_to_index.keys()
        x_ls = x.split()

        i = 0

        while i < len(x_ls):
            if x_ls[i] not in words:
                del x_ls[i]
            else:
                i += 1

        
        utf8_decoder = lambda x: x.decode('utf-8')

        # Decode it back to the string type
        x_ls = list(map(utf8_decoder, x_ls)) # Type str
        x = ' '.join(x_ls)

        return np.array(x, dtype = str)
    
    remover = np.vectorize(lambda x: remover_func(x, word_to_index))

    X = remover(X)

    return X
    



    






