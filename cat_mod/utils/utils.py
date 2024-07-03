import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import glorot_uniform
import os
import json
import shutil


def get_word_to_index(words: list | set, is_index_to_word: bool = False) -> dict | tuple:
    '''
    params:
        words (list or set): the list or set of words need to convert to index
        is_index_to_word (bool): if it is set to True it will be return with the word_to_index dict
        
    returns:
        word_to_index (dict): the dictionary with the translation word word to index
        
        OR
        
        word_to_index, index_to_word (dict, dict): if the is_index_to_word is set to True
        
    '''
    
    if (type(words) == set):
        words = sorted(list(words))
        
    word_to_index = {}
    index_to_word = {}
    
    
    for i in range(len(words)):
        word_to_index[words[i]] = i
        index_to_word[i] = words[i]
        
    if is_index_to_word:
        return word_to_index, index_to_word
    
    return word_to_index


def get_max_sentence_len(X: str) -> int:
    '''
    params:
        X (str): A training or testing or the whole dataset

    returns:
        max_sentence_len (int): A maximum length of the longest input sentence
    '''


    max_sentence_len = len(max(X, key = lambda x: len(x.split())).split())
    return max_sentence_len


def sentences_to_indices(X: np.array, word_to_index: dict, max_len: int) -> np.ndarray:
    """
    Description:
        Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
        The output shape should be such that it can be given to `Embedding()`
    
    params:
        X (np.array): array of sentences (strings), of shape (m,)
        word_to_index (dict): a dictionary containing the each word mapped to its index
        max_len (int): maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    returns:
    X_indices (np.array): array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    # Number of Training Examples
    m = X.shape[0]

    # Initialize the X_indices matrix
    X_indices = np.zeros((m, max_len))

    # Loop over each sentences in the dataset
    for i in range(m):

        # Get all the words in the ith sentence
        sentence_words = X[i].lower().split()

        # Loop over each word in the sentence
        # j is the index of the word in the sentence
        # w is the word itself in the sentence
        j = 0
        for w in sentence_words:
            if w in word_to_index.keys():
                X_indices[i, j] = word_to_index[w]
                j += 1

    return X_indices


def Y_to_indices(Y, is_idx_to_category = False, is_category_to_idx = False, count_unique = False):
    Y_unique = np.sort(np.unique(Y))

    category_to_idx = {}
    idx_to_category = {}

    for y_idx in range(len(Y_unique)):
        category_to_idx[Y_unique[y_idx]] = y_idx
        idx_to_category[y_idx] = Y_unique[y_idx]


    Y_indices = []
    for y in Y:
        Y_indices.append(category_to_idx[y])

    return_ls = []

    return_ls.append(np.array(Y_indices))

    if is_idx_to_category:
        return_ls.append(idx_to_category)
    
    if is_category_to_idx:
        return_ls.append(category_to_idx)

    if count_unique:
        return_ls.append(len(Y_unique))


    if len(return_ls) > 1:
        return return_ls
    
    return np.array(Y_indices)

def convert_to_one_hot(Y, num_of_row):
    Y = np.eye(num_of_row)[Y.reshape(-1)]
    return Y


def pretrained_embedding_layer(word_to_vec_map: dict, word_to_index: dict) -> tf.keras.layers.Embedding:
    '''
    Embedding matrix Size:

        ^
        |
        |
    vocab_size = len(word_to_index) + 1                 # Number of words in the dictionary
        |
        |
        ---------------------------------->
        emb_dim = word_to_vec_map[anyword].shape[0]     # The length of the vector of the word
                = 50                                    # GloVe dimension


        ^
        |
        |
        |
      words
        |
        |
        |
        |-------------- Categories -------------->
    '''

    vocab_size = len(word_to_index) + 1
    anyword = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[anyword].shape[0]

    # Initialize the embedding matrix
    emb_matrix = np.zeros((vocab_size, emb_dim))

    # Each row in the matrix represent the according word to its index
    # We will fill the matrix using the word_to_vec_map dictionary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # Define the Keras Embedding Layer Intances
    embedding_layer = Embedding(input_dim = vocab_size, output_dim = emb_dim, trainable = False)

    embedding_layer.build((None,))

    # Assign the pretrained matrix to the Keras Layer
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def Categorical_Model(input_shape, word_to_vec_map, word_to_index, num_of_category, num_of_LSTM = 2):

    '''
    
    sentence_indices -> embeddings -> LSTM -> Dropout -> LSTM -> Dropout -> Dense(softmax)
    
    
    '''


    sentence_indices = Input(shape = input_shape, dtype = 'int32')

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    embeddings = embedding_layer(sentence_indices)

    X = embeddings

    for i in range(num_of_LSTM):

        X = LSTM(units = 128, return_sequences = True)(X)

        X = Dropout(rate = 0.5)(X)

    # X = LSTM(units = 128, return_sequences = True)(embeddings)

    # X = Dropout(rate = 0.5)(X)

    # X = LSTM(units = 128, return_sequences = True)(embeddings)

    # X = Dropout(rate = 0.5)(X)

    # X = LSTM(units = 128, return_sequences = True)(embeddings)

    # X = Dropout(rate = 0.5)(X)

    # X = LSTM(units = 128, return_sequences = True)(embeddings)

    # X = Dropout(rate = 0.5)(X)

    # X = LSTM(units = 128, return_sequences = True)(embeddings)

    # X = Dropout(rate = 0.5)(X)

    # X = LSTM(units = 128, return_sequences = True)(embeddings)

    # X = Dropout(rate = 0.5)(X)


    X = LSTM(units = 128)(X)

    X = Dropout(rate = 0.5)(X)

    X = Dense(units = num_of_category)(X)

    X = Activation('softmax')(X)

    model = Model(inputs = sentence_indices, outputs = X)

    return model

def one_hot_to_category(Y_oh: np.array, idx_to_category: dict):
    
    result_ls = []

    # Number of Examples
    m = Y_oh.shape[0]

    for i in range(m):
        category_idx = np.argmax(Y_oh[i])
        result_ls.append(idx_to_category[category_idx])

    return np.array(result_ls)



def from_X_to_Y_predict(X: list, Y_dict: dict, model: tf.keras.Model, word_to_index: dict, max_len: int):
    
    

    if type(X) == str:
        X = np.array([X], dtype = str)

    # if type(X) == pd.core.series.Series:
    #     X = list(X)

    # if type(X) == list or type(X) == tuple:
    #     X = np.array(X)
    else:
        X = np.array(X, dtype = str)

    X_to_indices = sentences_to_indices(X, word_to_index, max_len)

    Y_predict_one_hot = model.predict(X_to_indices)

    Y_predict_category = one_hot_to_category(Y_predict_one_hot, Y_dict)

    return Y_predict_category


def export_model_data(file_name, catmod):
    dest_folder = file_name + '/'


    os.system('mkdir ' + file_name)
    file_name = file_name + '/' + file_name

    with open(file_name + '.weights.h5.txt', 'w') as f:
            f.writelines(str(catmod.MAX_STRING_LEN) + '\n')
            f.writelines(str(catmod.num_of_categories) + '\n')
            f.writelines(str(catmod.num_of_LSTM))


    with open(file_name + '.weights.h5.json', 'w') as f:
            json_object = json.dumps(catmod.index_to_category)
            f.write(json_object)

    print('Saving weights', end = '                         \r')
    catmod.model.save_weights(file_name + '.weights.h5')

    print('Exporting GloVe File...', end = '                        \r')
    shutil.copyfile(catmod.glove_file, dest_folder + 'glove.txt')


    print('Saved!', end = '                            \r')
    
def import_model_data(weights_file, catmod):
    dest_folder = weights_file + '/'
    weights_file = weights_file + '/' + weights_file
    


    with open(weights_file + '.weights.h5.txt', 'r') as f:
        catmod.MAX_STRING_LEN = int(f.readline())
        catmod.num_of_categories = int(f.readline())
        catmod.num_of_LSTM = int(f.readline())

    with open(weights_file + '.weights.h5.json', 'r') as f:
        temp_dict = json.load(f)
        result_dict = {}

        for key, value in temp_dict.items():
            result_dict[int(key)] = value
            
        catmod.index_to_category = result_dict

    catmod.load_model()

    print('Loading Weight...', end = '                           \r')
    

    catmod.model.load_weights(weights_file + '.weights.h5')
    print('Loaded!', end = '                               \r')
