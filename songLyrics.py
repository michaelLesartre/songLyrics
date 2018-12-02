import numpy
import pandas as pd
import tensorflow.keras as keras
from keras import Model, Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Reshape, Input, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
#https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

lyrics = pd.read_csv('lyrics.csv', engine='python', dtype = {'lyrics': str})

class Genre(object):
    def __init__(self, lyrics, name):
        self.lyrics = lyrics
        self.raw_text = '\n\n\n'.join(str(x) for x in lyrics).lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_to_int = dict((c, i) for i, c in enumerate(self.chars))
        self.dataX = []
        self.dataY = []
        self.n_patterns = 0
        self.model = None
        self.genre = name
        self.file_path = self.genre + "-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        self.y = None
        self.X = None

    def prepare_model(self, seq_length):
        for i in range(0, len(self.raw_text) - seq_length, 1):
            seq_in = self.raw_text[i:i + seq_length]
            seq_out = self.raw_text[i + seq_length]
            self.dataX.append([self.char_to_int[char] for char in seq_in])
            self.dataY.append(self.char_to_int[seq_out])
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
        genres[genre].prepare_model(10)

genres['Pop'].train_model()
