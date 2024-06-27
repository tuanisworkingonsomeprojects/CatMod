from utils.checking_dependencies import *
# Checking System Section
if __name__ != '__main__':
    checking_dependencies()
    

# Importing library section
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.utils import *
from utils.load_and_process_utils import *



# Main Class
class CatMod:

    

    MAX_STRING_LEN = 0
    
    log_status = False
    log_end_line = False

    X, Y, X_train, Y_train, X_test, Y_test, X_train_idx, Y_train_idx, Y_train_oh, X_test_idx, Y_test_idx, Y_test_oh = None, None, None, None, None, None, None, None, None, None, None, None

    dataset = None

    words = None
    word_to_vec_map = None

    word_to_index = None
    index_to_word = None

    index_to_category = None

    num_of_categories = 0


    
    def __init__(self, glove_file_path: str, model: tf.keras.Model = None, log_status: bool = False) -> None:
        self.model = model
        self.log_status = log_status
        print('Loading word to vector map...', end = '                               \r')
        self.words, self.word_to_vec_map = read_glove_vec(glove_file_path)

        print('Creating word to index dictionary...', end = '                          \r')
        self.word_to_index, self.index_to_word = get_word_to_index(self.words, is_index_to_word = True)


        print('CatMod instance created!                 ')


    def load_csv(self, file_path: str, X_column_name: str, Y_column_name: str, dropna = True):

        print('Importing csv file...', end = '                                     \r')
        self.dataset = get_dataset_from_csv(file_path, X_column_name, Y_column_name, dropna)

        print('Spliting Datset to X and Y...', end = '                            \r')
        self.X, self.Y = dataset_to_XY(self.dataset)

        print('Get the longest string in X (based on number of words)...', end = '                         \r')
        self.MAX_STRING_LEN = get_max_sentence_len(self.X)

        print('Creating X, Y training and testing dataset...', end = '                                      \r')
        self.X_train, self.Y_train, self.X_test, self.Y_test = get_train_test_dataset(self.dataset, to_XY = True)

        print('Indexing X_train and X_test dataset...', end = '                                 \r')
        self.X_train_idx = sentences_to_indices(self.X_train, self.word_to_index, max_len = self.MAX_STRING_LEN)
        self.X_test_idx = sentences_to_indices(self.X_test, self.word_to_index, max_len = self.MAX_STRING_LEN)

        print('Indexing Y_train and Y_test dataset...', end = '                                  \r')
        self.Y_train_idx, self.index_to_category, self.num_of_categories = Y_to_indices(self.Y_train, is_idx_to_category = True, count_unique = True)
        self.Y_test_idx = Y_to_indices(self.Y_test)

        print('Encoding One Hot Vectors for Y_train and Y_test...', end = '                               \r')
        self.Y_train_oh = convert_to_one_hot(self.Y_train_idx, self.num_of_categories)
        self.Y_test_oh = convert_to_one_hot(self.Y_test_idx, self.num_of_categories)


    def load_model(self):
        print('Loading model...', end = '                          \r')
        self.model = Categorical_Model((self.MAX_STRING_LEN,), self.word_to_vec_map, self.word_to_index, self.num_of_categories)

        print('Compiling model...', end =  '                       \r')
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Model Compiled Successfully!', end = '                       \r')

    def get_model_summary(self):
        print(self.model.summary())

    def load_weights(self, weights_file: str) -> None:
        print('Loading Weight...', end = '                           \r')
        self.model.load_weights()
        print('Loaded!', end = '                               \r')

    def train(self, epochs = 50):
        print('Training model...', '                          \r')
        self.model.fit(self.X_train_idx, self.Y_train_oh, epochs = epochs, batch_size = 32, shuffle=True)

    def save_weights(self, file_name: str):
        print('Saving weights', end = '                         \r')
        self.model.save_weights(file_name + '.weights.h5')
        print('Saved!', end = '                            \r')


    def evaluate(self):
        loss, acc = self.model.evaluate(self.X_test_idx, self.Y_test_oh)
        print()
        print("Test accuracy = ", acc)


    def predict(self, X):
        print('Predicting...')
        return_Y = from_X_to_Y_predict(X, self.index_to_category, self.model, self.word_to_index, self.MAX_STRING_LEN)
        return return_Y
    
    def print_test(self):
        print(__name__)
    
    
