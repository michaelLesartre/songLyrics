import numpy as np
import pandas as pd
import tensorflow.keras as keras
from keras import Model, Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Reshape, Input, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models import Word2Vec
import os

#https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

class Genre(object):
    def __init__(self, lyrics, name):
        self.lyrics = lyrics
        self.processed_sentences = [[word.strip(',.!') for word in x.split()] for x in lyrics]
        self.dataX = []
        self.dataY = []
        self.n_patterns = 0
        self.model = None
        self.name = name
        self.checkpoint_path = self.name + "-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        self.w2v_path = self.name+"_word2vec.model"
        self.y = None
        self.X = None

    def generate_results(self, num_words):
        seq_length = 25
        w2v = Word2Vec.load(self.w2v_path)
        word_vectors = w2v.wv

        max_checkpoint = 0
        best_checkpoint = None
        for checkpoint in os.listdir('.'):
            if self.name + "-weights-improvement" in checkpoint:
                num = int(checkpoint.split('-')[3])
                if num>max_checkpoint:
                    max_checkpoint=num
                    best_checkpoint=checkpoint

        print(f'best checkpoint found: {best_checkpoint}')
        self.model = Sequential()
        self.model.add(LSTM(256, input_shape=(seq_length, 100)))
        self.model.add(Dropout(.2))
        self.model.add(Dense(100, activation='softmax'))

        self.model.load_weights(best_checkpoint)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

        song_idx = np.random.randint(0, len(self.processed_sentences)-1)
        song = self.processed_sentences[song_idx]
        pattern = np.array([word_vectors.get_vector(word) for word in song[:seq_length]])
        for i in range(num_words):
            x=np.reshape(pattern, (1, seq_length, 100))
            prediction = self.model.predict(x)
            result = word_vectors.similar_by_vector(np.reshape(prediction, (100,)), topn=1)
            pattern = np.append(pattern, prediction, axis=0)
            pattern = pattern[1:len(pattern)]
            print(f'word: {result}')




    def prepare_model(self):
        seq_length=25
        w2v = Word2Vec(self.processed_sentences, size=100, window=5, min_count=1, workers=4)
        word_vectors = w2v.wv
        w2v.save(self.w2v_path)
        del w2v
        print("created word2vec model")


        total_songs = len(self.processed_sentences)
        processed_songs = 0

        for song in self.processed_sentences:
            if processed_songs % 25 == 0:
                print(f'Processed {processed_songs}/{total_songs} songs', end='\r')
            processed_songs+=1;
            for i in range(0, len(song) - seq_length, 1):
                seq_in = song[i:i + seq_length]
                seq_out = song[i+seq_length]
                self.dataX.append([word_vectors.get_vector(word) for word in seq_in])
                self.dataY.append(word_vectors.get_vector(seq_out))
        print(f'Processed {processed_songs}/{total_songs} songs')
        self.n_patterns = len(self.dataX)
        del word_vectors
        self.X = np.array(self.dataX)
        self.y = np.array(self.dataY)

        self.model = Sequential()
        self.model.add(LSTM(256, input_shape=(seq_length, 100)))
        self.model.add(Dropout(.2))
        self.model.add(Dense(100, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def train_model(self, epochs=20, batch_size=128):
        if not self.model:
            raise Exception('prepare model before training it')
        checkpoint = ModelCheckpoint(
            self.checkpoint_path,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(self.X, self.y,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=callbacks_list
                       )
