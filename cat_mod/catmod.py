from .utils.checking_dependencies import *
# Checking System Section
if __name__ != '__main__':
    checking_dependencies()
    

# Importing library section
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import tensorflow as tf
from .utils.utils import *
from .utils.load_and_process_utils import *
import os
import json
from .utils.cleaning import *
from .utils.data_pipeline import *


# Main Class
class CatMod:

    

    MAX_STRING_LEN: int = 0
    glove_file: str = None
    
    X: np.ndarray = None # shape: (number_of_examples, )
    Y: np.ndarray = None # shape: (number_of_examples, )
    X_train: np.ndarray = None # shape: (number_of_training_examples, )
    Y_train: np.ndarray = None # shape: (number_of_training_examples, )
    X_test: np.ndarray = None  # shape: (number_of_testing_examples, )
    Y_test: np.ndarray = None  # shape: (number_of_testing_examples, )
    X_train_idx: np.ndarray = None # shape: (number_of_training_examples, MAX_STRING_LEN)
    Y_train_idx: np.ndarray = None # shape: (number_of_training_examples, )
    Y_train_oh: np.ndarray = None  # shape: (number_of_training_examples, number_of_categories)
    X_test_idx: np.ndarray = None  # shape: (number_of_testing_examples, MAX_STRING_LEN)
    Y_test_idx: np.ndarray = None  # shape: (number_of_testing_examples, )
    Y_test_oh: np.ndarray = None   # shape: (number_of_testing_examples, number_of_categories)

    dataset: pd.DataFrame = None

    words: list | set = None
    word_to_vec_map: dict = None

    word_to_index: dict = None
    index_to_word: dict = None

    index_to_category: dict = None

    num_of_categories: int = 0

    num_of_LSTM: int = 2


    
    def __init__(self, glove_file_path: str = None, model: tf.keras.Model = None, load_mode = False, load_file = None) -> None:
        '''
        Description:
            This is the constructor of the CatMod Class which will need the Word_to_Vec Embedding File as an input
            If user have their own pre-defined model they can load it in by passing it as the parameter in this constructor

        Params:
            glove_file_path (str) (compulsory): Word_to_Vec file path. It is not necessary to be a GloVe file but any valid file
            model (tf.keras.Model) (optional): user pre-defined model structure / architechture

        Returns:
            None
        '''

        if load_mode:
            assert load_file != None, "Please provide a load file name"
            self.glove_file = load_file + '/glove.txt'
        else:
            assert glove_file_path != None, "Please provide a GloVe file name"
            self.glove_file = glove_file_path

        self.model = model

        print('Loading word to vector map...', end = '                               \r')
        self.words, self.word_to_vec_map = read_glove_vec(self.glove_file)

        print('Creating word to index dictionary...', end = '                          \r')
        self.word_to_index, self.index_to_word = get_word_to_index(self.words, is_index_to_word = True)

        if load_mode:
            self.load_weights(load_file)

        print('CatMod instance created!                 ')



    def get_shape(self):
        print('X:', self.X.shape)
        print('Y:', self.Y.shape)
        print('X_train:', self.X_train.shape)
        print('Y_train:', self.Y_train.shape)
        print('X_train_idx:', self.X_train_idx.shape)
        print('Y_train_idx:', self.Y_train_idx.shape)
        print('Y_train_oh:', self.Y_train_oh.shape)
        print('X_test:', self.X_test.shape)
        print('Y_test:', self.Y_test.shape)
        print('X_test_idx:', self.X_test_idx.shape)
        print('Y_test_idx:', self.Y_test_idx.shape)
        print('Y_test_oh:', self.Y_test_oh.shape)





    def load_csv(self, file_path: str, X_column_name: str, Y_column_name: str, dropna = True) -> None:
        '''
        Description:
            This is the method that allows user to load their csv dataset file with value and target column labeled accordingly
            The User have to specify which column represents the X / value dataset and which represents the Y / target dataset

        Params:
            file_path (str) (Compulsory): the file path of leads to the csv file
            X_column_name (str) (Compulsory): the name of the X / value column in the csv file
            Y_column_name (str) (Compulsory): the name of the Y / targe column in the csv file
            dropna (bool) (Optional): it is recommended to leave it to be True as it is better for the model to train on the clean dataset

        Returns:
            None
        '''

        print('Importing csv file...', end = '                                     \r')
        self.dataset = get_dataset_from_csv(file_path, X_column_name, Y_column_name, dropna)

        print('Spliting Datset to X and Y...', end = '                            \r')
        self.X, self.Y = dataset_to_XY(self.dataset)











        print('Going under pipeline...', end = '                                             \r')
        self.X = pipeline(self.X, training_predicting = 'training', word_to_index = self.word_to_index, preprocess = True)












        print('Get the longest string in X (based on number of words)...', end = '                         \r')
        self.MAX_STRING_LEN = get_max_sentence_len(self.X)


        print('Creating X, Y training and testing dataset...', end = '                                      \r')
        self.X_train, self.Y_train, self.X_test, self.Y_test = get_train_test_dataset(self.dataset, to_XY = True)

        # print('Indexing X_train and X_test dataset...', end = '                                 \r')
        # self.X_train_idx = sentences_to_indices(self.X_train, self.word_to_index, max_len = self.MAX_STRING_LEN)
        # self.X_test_idx = sentences_to_indices(self.X_test, self.word_to_index, max_len = self.MAX_STRING_LEN)


        print('Indexing X_train and X_test dataset...', end = '                                 \r')
        self.X_train_idx = pipeline(self.X_train, 'training', self.word_to_index, self.MAX_STRING_LEN)
        self.X_test_idx  = pipeline(self.X_test, 'training', self.word_to_index, self.MAX_STRING_LEN)


        print('Indexing Y_train and Y_test dataset...', end = '                                  \r')
        self.Y_train_idx, self.index_to_category, self.num_of_categories = Y_to_indices(self.Y_train, is_idx_to_category = True, count_unique = True)
        self.Y_test_idx = Y_to_indices(self.Y_test)

        print('Encoding One Hot Vectors for Y_train and Y_test...', end = '                               \r')
        self.Y_train_oh = convert_to_one_hot(self.Y_train_idx, self.num_of_categories)
        self.Y_test_oh = convert_to_one_hot(self.Y_test_idx, self.num_of_categories)


    def load_model(self, csv_file = None, X_column = None, Y_column = None, num_of_LSTM = 2):
        '''
        Description:
            This function will load the pre-defined model in the utils.Categorical_Model into the instance
        
        Params:
            None

        Returns:
            None
        '''
        self.num_of_LSTM = num_of_LSTM

        if csv_file != None:
            self.load_csv(csv_file, X_column, Y_column)

        print('Loading model...', end = '                                                                 \r')
        self.model = Categorical_Model((self.MAX_STRING_LEN,), self.word_to_vec_map, self.word_to_index, self.num_of_categories, num_of_LSTM = num_of_LSTM)

        print('Compiling model...', end =  '                                                          \r')
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Model Compiled Successfully!', end = '                                                   \r')

    def get_model_summary(self):
        '''
        Description:
            Print the model summary

        Params:
            None

        Returns:
            None
        '''

        print(self.model.summary())

    def load_weights(self, weights_file: str) -> None:
        '''
        Description:
            This method will allow user to load the pre-trained parameters into the model
            Note: But the pre-trained parametered must be trained on the same architecture with one that defined in utils.Categorical_Model

        Params:
            weights_file (str) (Compulsory): The file path of the pre-trained weight

        Returns:
            None
        
        '''
        import_model_data(weights_file, self)




    def train(self, epochs: int = 50) -> None:
        '''
        Description:
            This method will start to train the model with the given dataset
            The dataset must be loaded before training

        Params:
            epochs (int) (Optional): User desired number of iterations (default is 50)
        
        Returns:
            None
        '''


        print('Training model...', '                          \r')
        self.model.fit(self.X_train_idx, self.Y_train_oh, epochs = epochs, batch_size = 32, shuffle=True)

    def save_weights(self, file_name: str) -> None:
        '''
        Description:
            This method allow user to export their trained parameters to a desired file name

        Params:
            file_name (str) (Compulsory): The name of the exported file (DO NOT include the file suffix)

        Returns:
            None
        '''
        export_model_data(file_name, self)


    def evaluate(self) -> None:
        '''
        Description:
            This method will evaluated the trained model with the test dataset extracted from the imported dataset

        Params:
            None

        Returns:
            None
        '''

        loss, acc = self.model.evaluate(self.X_test_idx, self.Y_test_oh)
        print()
        print("Test accuracy = ", acc)


    def predict(self, X: list | str, to_df = False, to_csv = False, to_excel = False, file_name = None, X_name = 'X', Y_name = 'Y') -> np.ndarray:
        '''
        Desctiption:
            This method will allow the model to predict the provided string or list of string and return the appropriate predicted category / categories

        Params:
            X (list or str) (Compulsory): The input string or list of string you want to predict the category

        Returns:
            Y (np.ndarray): The prediction according to the input string
        
        '''


        print('Predicting...')
        X_idx = pipeline(X, training_predicting = 'predicting', word_to_index = self.word_to_index, max_len = self.MAX_STRING_LEN, preprocess = False)
        return_Y = from_X_to_Y_predict(X_idx, self.index_to_category, self.model, self.word_to_index, self.MAX_STRING_LEN)
        

        if to_df:
            return pd.DataFrame({
                X_name: X,
                Y_name: return_Y
            })
        
        if to_csv:
            return pd.DataFrame({
                X_name: X,
                Y_name: return_Y
            }).to_csv(file_name + '.csv', index = False)

        if to_excel:
            return pd.DataFrame({
                X_name: X,
                Y_name: return_Y
            }).to_csv(file_name + '.csv').to_excel(file_name + '.xlsx', index = False)

        return return_Y
    

    
    def print_test(self):
        print()
        export_model_data("asdfasdf",self)
    
