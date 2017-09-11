import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import LSTM
import keras
import re

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(0,len(series), window_size):
        if i+window_size < len(series):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size])
   
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    rnnModel = Sequential()
    # incraese the hidden unit to 10 from 5 in the instuction to improve the accuracy
    rnnModel.add(LSTM(10, input_shape=(window_size,1)))
    rnnModel.add(Dense(1))
    return rnnModel

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    chars = list(text)

    def isAlphaNumericOrSpace(c):
        return re.match('\w',str(c)) or c == ' '   

    texts = list(filter(lambda x: isAlphaNumericOrSpace(x) or x in punctuation, chars))
    return ''.join(texts)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text), step_size):
        if i+window_size < len(text):
            inputs.append(text[i:i+window_size])
            outputs.append(text[i+window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    rnnModel = Sequential()
    rnnModel.add(LSTM(200, input_shape=(window_size,num_chars)))
    rnnModel.add(Dense(num_chars, activation='linear'))
    rnnModel.add(Activation('softmax'))
    return rnnModel    
