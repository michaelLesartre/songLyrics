import numpy as np
import pandas as pd
import tensorflow.keras as keras
from keras import Model, Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Reshape, Input, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models import Word2Vec
import os

class genre(object):
    def __init__(self, lyrics, name):
        self.lyrics = lyrics
        self.processed_sentences = [[word.strip('{}[](),.!') for word in x.split() if word!='[Chorus]'] for x in lyrics][:7000]
        self.dataX = None
        self.dataY = None
        self.n_patterns = 0
        self.model = None
        self.name = name
        self.checkpoint_path = self.name + "-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        self.w2v_path = self.name+"_word2vec.model"
        self.y = None
        self.X = None
        self.seq_length = 14
        self.wv_len = 100

    def generate_results(self, num_words):
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

        if not best_checkpoint:
            raise FileNotFoundError(f'no weights for {self.name} found. Has this model been trained?')
        print(f'best checkpoint found: {best_checkpoint}')
        self.model = Sequential()
        self.model.add(LSTM(256, input_shape=(self.seq_length, self.wv_len)))
        self.model.add(Dropout(.1))
        self.model.add(Dense(self.wv_len, activation='tanh'))

        self.model.load_weights(best_checkpoint)
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        song_idx = np.random.randint(0, len(self.processed_sentences)-1)
        song = self.processed_sentences[song_idx]
        while len(song)<self.seq_length:
            song_idx = np.random.randint(0, len(self.processed_sentences)-1)
            song = self.processed_sentences[song_idx]

        pattern = np.array([word_vectors.get_vector(word) for word in song[:self.seq_length]])
        song = ""
        for i in range(num_words):
            x=np.reshape(pattern, (1, self.seq_length, self.wv_len))
            prediction = self.model.predict(x)
            options = word_vectors.similar_by_vector(np.reshape(prediction, (self.wv_len,)), topn=5)
            option_idx = np.random.randint(0, 4)
            result = options[option_idx][0]
            pattern = np.append(pattern, [word_vectors[result]], axis=0)
            pattern = pattern[1:len(pattern)]
            song += result
            song += " "
        print(song)



    def prepare_model(self):
        w2v = Word2Vec(self.processed_sentences, size=self.wv_len, window=5, min_count=1, workers=4)
        word_vectors = w2v.wv
        w2v.save(self.w2v_path)
        del w2v
        print("created word2vec model")
        n_patterns = 0
        for song in self.processed_sentences:
            for i in range(0, len(song) - self.seq_length, 1):
                n_patterns+=1
        self.dataX = np.zeros((n_patterns, self.seq_length, self.wv_len))
        self.dataY = np.zeros((n_patterns, self.wv_len))

        total_songs = len(self.processed_sentences)
        processed_songs = 0

        pattern = 0
        for song in self.processed_sentences:
            if pattern % 25 == 0:
                print(f'Processed {pattern}/{n_patterns} patterns', end='\r')
            processed_songs+=1;
            for i in range(0, len(song) - self.seq_length, 1):
                seq_in = song[i:i + self.seq_length]
                seq_out = song[i+self.seq_length]
                self.dataX[pattern] =  [word_vectors.get_vector(word) for word in seq_in]
                self.dataY[pattern] = word_vectors.get_vector(seq_out)
                pattern += 1
        print(f'Processed {pattern}/{n_patterns} patterns')
        del word_vectors

        self.model = Sequential()
        self.model.add(LSTM(256, input_shape=(self.seq_length, self.wv_len)))
        self.model.add(Dropout(.1))
        self.model.add(Dense(self.wv_len, activation='tanh'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

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
        self.model.fit(self.dataX, self.dataY,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=callbacks_list
                       )
