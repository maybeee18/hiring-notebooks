'''
RNN suite for fitting time series data
'''

from __future__ import print_function
from __future__ import division

__version__ = "1.3.0"
__author__ = "Vishnu Dutt Sharma"
__email__ = "Vishnu.D.Sharma3@aexp.com"

import os
os.environ['PATH'] = ':'.join(['/usr/bin/','/usr/local/cuda/bin/',os.environ['PATH']])
os.environ["THEANO_FLAGS"] = "device=cuda,floatX=float32"
try:
    os.environ['LD_LIBRARY_PATH'] = ':'.join(['/usr/local/cuda/lib64', os.environ['LD_LIBRARY_PATH']])
except KeyError:
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'

import sys
import numpy as np
import pandas as pd
import json
import time
import argparse
from keras.models import Model, load_model
from keras.layers import Input, SimpleRNN, GRU, LSTM, merge, Dense, Embedding, multiply, Dropout, Activation, Reshape, concatenate, add, Merge
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.constraints import unit_norm
from keras.models import load_model
import keras.backend as K
if (K.backend() == 'tensorflow'):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(1)
import random
random.seed(1)

def argsListGen():
    """Method for parsing command-line arguments.

    Options:
        Running mode: `data_prep`, `train`, `test`. Atleast one mode should be provided
        Config location:  Path of the config(json) file

    Returns:
        Parsed object to be used as a dictionary.

    """
    ap = argparse.ArgumentParser(description='--data_prep: Data preparation mode \n --train: Training mode \n --test: Prediction mode')
    ap.add_argument('--data_prep', action="store_true", dest="data_prep", default=False)
    ap.add_argument('--build', action="store_true", dest="build", default=False)
    ap.add_argument('--score', action="store_true", dest="score", default=False)

    ap.add_argument('--config', action="store", dest="config", default="./config.json")

    arg = ap.parse_args()
    return arg


def get_config(filename):
    """Reads configuration form file.

    Returns:
        {
            `config`: Returns dictionary from a json file containing configuration parameters.
        }

    """
    return json.loads(open(filename,'r').read())


class DataClass():
    """Class for data processing.

    Args:
        config (:obj:`dict`): Dictionary containing configuration parameters for
            data preparation step.

    """
    def __init__(self, config):
        self.keyname = config['keyname']
        self.def_ind = config['def_ind']
        self.ndim    = config['num_dim']
        self.maxlen  = config['maxlen']

        self.config  = config

        self.usecol = [self.keyname, self.def_ind]
        for i in range(self.ndim):
            self.usecol.append( config['name_dim'+str(i+1)] )

    def pad_arr(self, txns, maxlen):
        """Sequence padding/truncation function.

        On the basis of sequence length, truncates or pads(with 0s) input array.

        Args:
            txns (:obj:`list` of :obj:`int`): Array of integers(a sequence)
            maxlen (int): Desired length of sequence.

        Returns:
            2-D Numpy array of int with 'maxlen' number of columns.

        """
        data = np.zeros((maxlen))
        if(len(txns) >= maxlen):
            data = np.array(txns[-maxlen:])
        else:
            data[-len(txns):] = np.array(txns)
        return data

    def prep_data(self, filename):
        """Data preparation method.

        Performs a key-level groupby, calls `data_prep` method to create sequences.

        Args:
            filename (str): location of raw data path.

        """
        start_time=time.time()

        data_df = pd.read_csv(filename, usecols=self.usecol)
        data_df[self.keyname] = data_df[self.keyname].astype(str)

        gpby_obj = data_df.groupby(self.keyname)

        def_dict = {k: int(np.sum(list(v))>0) for k, v in gpby_obj[self.def_ind]}
        data_dict = {}
        for nm in self.usecol[2:]:
            data_dict[nm] = {k: self.pad_arr(v, self.maxlen) for k, v in gpby_obj[nm]}

        num_cust = len(def_dict.keys())

        self.customer  = ['']*num_cust
        self.y_train   = np.zeros(shape=(num_cust))
        self.X_train = []
        for i in range(self.ndim):
            self.X_train.append(np.zeros(shape=(num_cust, self.maxlen), dtype=np.int32))

        key_list = list(def_dict.keys())
        for cnt in range(num_cust):
            key = key_list[cnt]
            self.customer[cnt] = key
            for i in range(self.ndim):
                self.X_train[i][cnt] = data_dict[self.config['name_dim'+str(i+1)]][key]
                self.y_train[cnt] = def_dict[key]

        print("Time taken for Preparing Data : " , str(time.time() - start_time)  , " in sec.")

    def save_csv(self, filename):
        new_df = pd.DataFrame()
        new_df[self.keyname] = self.customer
        new_df[self.def_ind]  = self.y_train
        for dim in range(self.ndim):
            for tm in range(self.maxlen):
                new_df[ self.config['name_dim'+str(dim+1)]+'_'+str(tm)] = self.X_train[dim][:,tm]
        new_df.to_csv(filename, index=False)

    def load_from_csv(self, filename):
        """Loads data form a CSV containing processed data.

        Args:
            filename (str): Location of the file containing processed data.
        """
        start_time=time.time()

        new_df = pd.read_csv(filename)
        self.customer = new_df[self.keyname]
        self.y_train  = new_df[self.def_ind]

        num_cust = len(self.customer)
        self.X_train = []
        for i in range(self.ndim):
            self.X_train.append(np.zeros(shape=(num_cust, self.maxlen), dtype=np.int32))

        for dim in range(self.ndim):
            for tm in range(self.maxlen):
                self.X_train[dim][:,tm] = new_df[ self.config['name_dim'+str(dim+1)]+'_'+str(tm) ]

        print("Time taken for Reading Data : " , str(time.time() - start_time)  , " in sec.")

    def get_data(self):
        """Getter for input data and label.


        Returns:
            {
                `X_train`: numpy array containing input sequences.
                `y_train`: numpy array of training labels.
            }
        """
        return (self.X_train, self.y_train)

    def get_keys(self):
        """Getter for keys in data

        Returns:
            {
                `customer`: list of keys in same order as training data.
            }
        """
        return self.customer


class ModelClass():
    """Class for RNN model.

    Args:
        config (:obj:`dict`): Dictionary containing configuration parameters for
            model builing step.

    """
    def __init__(self, config):
        self.ndim = config['num_dim']
        self.maxlen = config['maxlen']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.rnn = None
        try:
            self.arch = config['arch']
        except KeyError:
            self.arch = 'add'


    def make_from_config(self, config):
        """Builds model using configurations given in model file.

        Args:
            config (:obj:`dict`): Dictionary containing configuration parameters.

        """
        start_time=time.time()

        input_layers = [None]*self.ndim
        embd_layers  = [None]*self.ndim
        for i in range(self.ndim):
            input_layers[i] = Input(shape=(self.maxlen,), name='INP_'+config['name_dim'+str(i+1)])
            X = Embedding(input_dim=config['max_feat_dim'+str(i+1)] + 1, output_dim=config['embed_size_dim'+str(i+1)], name='EMBD_'+config['name_dim'+str(i+1)])(input_layers[i])
            embd_layers[i] =Dropout(rate=config['dropout_dim'+str(i+1)])(X)

        if self.ndim > 1:
            if self.arch == 'mult' or self.arch == 'multiply':
                C = [None]*self.ndim
                if config['rnn_type'] == 'SimpleRNN':
                    for i in range(self.ndim):
                        C[i] = SimpleRNN(config['rnn_output_size'], recurrent_dropout=config['rnn_rec_dropout'],dropout=config['rnn_dropout'])(embd_layers[i])
                elif config['rnn_type'] == 'GRU':
                    for i in range(self.ndim):
                        C[i] = GRU(config['rnn_output_size'], recurrent_dropout=config['rnn_rec_dropout'],dropout=config['rnn_dropout'])(embd_layers[i])
                else:
                    for i in range(self.ndim):
                        C[i] = LSTM(config['rnn_output_size'], recurrent_dropout=config['rnn_rec_dropout'],dropout=config['rnn_dropout'])(embd_layers[i])
                X = multiply(C)
            else:
                if self.arch == 'add':
                    C = concatenate(embd_layers)
                    C = Reshape(target_shape=(self.maxlen, config['embed_size_dim1'], self.ndim))(C)
                    X = TimeDistributed(Dense(units=1, input_shape=(config['embed_size_dim1'],self.ndim), use_bias=False, kernel_constraint=unit_norm() ))(C)
                    X = Dropout(rate=config['wgt_dense_drop'])(X)
                    X = Reshape(target_shape=(self.maxlen, config['embed_size_dim1'],))(X)
                elif self.arch == 'concat':
                    X = concatenate(embd_layers)

                if config['rnn_type'] == 'SimpleRNN':
                    X = SimpleRNN(config['rnn_output_size'], recurrent_dropout=config['rnn_rec_dropout'],dropout=config['rnn_dropout'])(X)
                elif config['rnn_type'] == 'GRU':
                    X = GRU(config['rnn_output_size'], recurrent_dropout=config['rnn_rec_dropout'],dropout=config['rnn_dropout'])(X)
                else:
                    X = LSTM(config['rnn_output_size'], recurrent_dropout=config['rnn_rec_dropout'],dropout=config['rnn_dropout'])(X)
        else:
            X = embd_layers[0]

        X = Dense(config['dense1_num_node'])(X)
        X = Dropout(rate=config['dense1_dropout'])(X)
        y = Dense(1, activation='sigmoid')(X)

        self.rnn = Model(inputs=input_layers, outputs=y)
        self.rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Time taken for Building model : " , str(time.time() - start_time)  , " in sec.")

    def make_from_json(self, filename):
        """Builds model from json file.

        Args:
            filename (str): Location of the standard keras architecture json file.

        """
        model_config = json.loads(open(filename,'r').read())
        self.rnn = model_from_yaml(model_config)
        self.rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train_model(self, x, y):
        """Trains the model.

        Args:
            x (:obj:`np.ndarray`): numpy array containing input data.
            y (:obj:`list` or `np.ndarray`): list or numpy array containing input labels.

        """
        start_time=time.time()
        self.rnn.fit(x=x, y=y, batch_size=self.batch_size, epochs=self.epochs, verbose=1)
        print("Time taken for training model : " , str(time.time() - start_time)  , " in sec.")

    def predict(self, x):
        """Makes predicitons from the model.

        Args:
            x (:obj:`np.ndarray`): numpy array containing input data.

        Returns:
            {
                `score`: `np.array` containing scores.
            }

        """
        start_time=time.time()
        score = self.rnn.predict(x=x, batch_size=self.batch_size, verbose=1)
        print("Time taken for prediction : " , str(time.time() - start_time)  , " in sec.")
        return score

    def print_model(self):
        """Prints the summary of the model.

        """
        print(self.rnn.summary())

    def get_model(self):
        """Getter for the model.

        """
        return self.rnn

    def save_model(self, filename):
        """Writes the output to a file.

        Args:
            filename (str): Location of the file where output is written out.
                Output will contain key, score and label.

        """
        self.rnn.save(filename, overwrite=True)

    def load_model(self, filename):
        self.rnn = load_model(filename)


def main():
    """Main executer code.

    Depending upon the mode and config parameters, it prapares/reads data,
    builds model, and/or scores the data and writes out to a file.
    """
    begin_time=time.time()

    arg = argsListGen()
    if not arg.data_prep and not arg.build and not arg.score:
        sys.exit("Please provide atleast one of the options: --data_prep, --build, --score")
    #print(arg)

    model_config = get_config(arg.config) #Load configuration

    x, y = None, None
    data_obj = None
    RNN_obj = None

    if arg.data_prep:
        ## Preparing Data
        data_obj = DataClass(model_config) #Create a data object with your configuration
        data_obj.prep_data(model_config['RAW_DATA_PATH']) #Prepare data
        data_obj.save_csv(model_config['PREP_DATA_PATH']) #Save data to a csv
        (x, y) = data_obj.get_data() #Get data

    if arg.build:
        if x is None or y is None:
            ## Loading from already prepared data
            data_obj = DataClass(model_config)
            data_obj.load_from_csv(model_config['PREP_DATA_PATH'])
            x,y = data_obj.get_data()

        ## Making RNN model
        RNN_obj = ModelClass(model_config)
        try:
            print('Loading model from {}'.format(model_config['PRETRAINED_MODEL_PATH']))
            RNN_obj.load_model(model_config['PRETRAINED_MODEL_PATH'])
        except KeyError:
            print('Building model from config file')
            RNN_obj.make_from_config(model_config)

        RNN_obj.print_model() #Print the model architecture

        RNN_obj.train_model(x=x, y=y)
        RNN_obj.save_model(model_config['MODEL_PATH']) #To save the model

    if arg.score:
        if x is None or y is None:
            ## Loading from already prepared data
            data_obj = DataClass(model_config)
            data_obj.load_from_csv(model_config['PREP_DATA_PATH'])
            x,y = data_obj.get_data()


        if RNN_obj == None:
            RNN_obj = ModelClass(model_config)

        # Prediction
        RNN_obj.load_model(model_config['MODEL_PATH']) #to load a saved model
        score = RNN_obj.predict(x) #Get predictions

        new_df = pd.DataFrame()
        new_df[model_config['keyname']] = data_obj.get_keys()
        new_df[model_config['def_ind']] = y
        new_df['score'] = score
        new_df.to_csv(model_config['SCORE_OUT_PATH'], index=False)

    print("Total Time taken from end to end: " , str(time.time() - begin_time)  , " in sec.")

if __name__ == "__main__":
    main()
