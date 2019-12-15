"""
Author: Jay Lux Ferro
Date:   12 Dec 2019
Models
"""

from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten, GRU, Activation, SimpleRNN
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
import my_layers as ll

def simpleRNN(num_features):
    model = Sequential()
    model.add(SimpleRNN(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(SimpleRNN(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

def simpleRNN2(num_features):
    model = Sequential()
    model.add(SimpleRNN(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(SimpleRNN(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

def lstm(num_features):
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

def lstm2(num_features):
    model = Sequential()
    model.add(LSTM(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(LSTM(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

def gru(num_features):
    model = Sequential()
    model.add(GRU(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(GRU(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

def gru2(num_features):
    model = Sequential()
    model.add(GRU(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(GRU(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model


def lstm_gru(num_features):
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(GRU(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

def lstm_gru2(num_features):
    model = Sequential()
    model.add(LSTM(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(GRU(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model


def gru_lstm(num_features):
    model = Sequential()
    model.add(GRU(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

def gru_lstm2(num_features):
    model = Sequential()
    model.add(GRU(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(LSTM(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model


def simpleRNN_lstm(num_features):
    model = Sequential()
    model.add(SimpleRNN(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

def simpleRNN_lstm2(num_features):
    model = Sequential()
    model.add(SimpleRNN(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(LSTM(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model


def lstm_simpleRNN(num_features):
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(SimpleRNN(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

def lstm_simpleRNN2(num_features):
    model = Sequential()
    model.add(LSTM(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(SimpleRNN(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model


def simpleRNN_gru(num_features):
    model = Sequential()
    model.add(SimpleRNN(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(GRU(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

def simpleRNN_gru2(num_features):
    model = Sequential()
    model.add(SimpleRNN(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(GRU(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model


def gru_simpleRNN(num_features):
    model = Sequential()
    model.add(GRU(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(SimpleRNN(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

def gru_simpleRNN2(num_features):
    model = Sequential()
    model.add(GRU(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, num_features], return_sequences=True))
    model.add(SimpleRNN(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model
