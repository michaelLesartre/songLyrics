import numpy
import pandas as pd
import tensorflow.keras as keras
from keras import Model, Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Reshape, Input, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import re
#https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

lyrics = pd.read_csv('lyrics.csv', engine='python', dtype = {'lyrics': str})

class Genre(object):
    def __init__(self, lyrics, name):
        self.lyrics = lyrics
        self.raw_text = '\n\n\n'.join(str(x) for x in lyrics).lower()
        self.processed_words = [word.strip(',.!') for word in self.raw_text.split()]
        self.words = sorted(list(set(self.processed_words)))
        self.word_to_int = dict((w, i) for i, w in enumerate(self.words))
        self.dataX = []
        self.dataY = []
        self.n_patterns = 0
        self.model = None
        self.genre = name
        self.file_path = self.genre + "-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        self.y = None
        self.X = None

    def prepare_model(self, seq_length):

        for i in range(0, len(self.processed_words) - seq_length, 1):
            seq_in = self.processed_words[i:i + seq_length]
            seq_out = self.processed_words[i + seq_length]
            self.dataX.append([self.word_to_int[word] for word in seq_in])
            self.dataY.append(self.word_to_int[seq_out])
        self.n_patterns = len(self.dataX)

        X = numpy.reshape(self.dataX, (self.n_patterns, seq_length, 1))
        self.X = X / float(len(self.chars))
        self.y = np_utils.to_categorical(self.dataY)

        self.model = Sequential()
        self.model.add(LSTM(256, input_shape=(self.X.shape[1], self.X.shape[2])))
        self.model.add(Dropout(.2))
        self.model.add(Dense(self.y.shape[1], activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def train_model(self, epochs=20, batch_size=64):
        if not self.model:
            raise Exception('prepare model before training it')
        checkpoint = ModelCheckpoint(self.file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

genres = {}

for genre in lyrics.genre.unique():
    if genre not in ['Not Available', 'Other']:
        genres[genre] = Genre(lyrics[lyrics['genre']==genre]['lyrics'], genre)

genres['Pop'].prepare_model(10)
genres['Pop'].train_model()
