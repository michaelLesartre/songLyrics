import numpy as np
import pandas as pd
import tensorflow.keras as keras
from keras import Model, Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Reshape, Input, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models import Word2Vec

#https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

lyrics = pd.read_csv('lyrics.csv', engine='python', dtype={'lyrics': str})

class Genre(object):
    def __init__(self, lyrics, name):
        self.lyrics = lyrics
        self.processed_sentences = [[word.strip(',.!') for word in x.split()] for x in lyrics][:300]
        self.raw_text = '\n\n\n'.join(str(x) for x in lyrics).lower()
        self.processed_words = [word.strip(',.!') for word in self.raw_text.split()]
        self.dataX = []
        self.dataY = []
        self.n_patterns = 0
        self.model = None
        self.name = name
        self.checkpoint_path = self.name + "-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        self.w2v_path = self.name+"_word2vec.model"
        self.y = None
        self.X = None

    def generate_results(self){
        w2v = Word2Vec.load(self.w2v_path)
    }

    def prepare_model(self, seq_length, wv_length=100):

        w2v = Word2Vec(self.processed_sentences, size=wv_length, window=5, min_count=1, workers=4)
        word_vectors = w2v.wv
        w2v.save(self.w2v_path)
        del w2v
        print("created word2vec model")


        total_songs = len(self.processed_sentences)
        processed_songs = 0

        for song in self.processed_sentences:
            if processed_songs % 100 == 0:
                print(f'Processed {processed_songs}/{total_songs} songs', end='\r')
            processed_songs+=1;
            for i in range(0, len(song) - seq_length, 1):
                seq_in = song[i:i + seq_length]
                seq_out = song[i+seq_length]
                self.dataX.append([word_vectors.get_vector(word) for word in seq_in])
                self.dataY.append(word_vectors.get_vector(seq_out))
        print()
        self.n_patterns = len(self.dataX)

        self.X = np.array(self.dataX)
        self.y = np.array(self.dataY)

        self.model = Sequential()
        self.model.add(LSTM(256, input_shape=(seq_length, wv_length)))
        self.model.add(Dropout(.2))
        self.model.add(Dense(wv_length, activation='softmax'))
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


genres = {}

for genre in lyrics.genre.unique():
    if genre not in ['Not Available', 'Other']:
        genres[genre] = Genre(lyrics[lyrics['genre'] == genre]['lyrics'], genre)

genres['Metal'].prepare_model(50)
print("model is prepared, starting training")
genres['Metal'].train_model()
