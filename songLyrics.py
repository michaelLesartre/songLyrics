import pandas as pd
import tensorflow as tf
import numpy as np
import re
from collections import defaultdict

# load in data
lyrics = pd.read_csv('lyrics.csv')
genre_lyrics = {}
word_p = re.compile(r'[a-zA-Z][^\s;,]*')

for genre in lyrics.genre.unique():
    if genre not in ['Not Available', 'Other']:
        genre_lyrics[genre] = lyrics[lyrics['genre']==genre]['lyrics']
        wordcounts = defaultdict(int)
        for song in genre_lyrics[genre]:
            if type(song) is str:
                for word in word_p.findall(song):
                    wordcounts[word.lower()] += 1

# return random data rows for testing
def get_next_batch(size, genre):
    return np.array([word_to_num(song) for song in genre_lyrics[genre].sample(size)])

# convert a string on song lyrics to string of numbers
def word_to_num(song):
    return [ord(letter) for letter in str(song).ljust(10000, '\x00')]

print(get_next_batch(128, 'Pop'))

def num_to_word(song):
    return "".join([chr(num) for num in song])

def xavier_init(size):
    in_dim = size[0]
    xavier_stdev = 1./tf.sqrt(in_dim/2.)
    return tf.random_normal(shape=size, stddev = xavier_stdev)

#Discriminator
X = tf.placeholder(tf.float32, shape=[None, 10000], name='X')


D_W1 = tf.Variable(xavier_init([10000, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

#Generator
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

G_W2 = tf.Variable(xavier_init([128, 10000]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[10000]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.relu(G_log_prob)

    return G_prob

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real)+ tf.log(1 - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

# initialize testing variables
train_size = 128
Z_dim = 100
current_genre = 'Pop'

# begin session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# run session iterations for training
for it in range(100000):
    if it % 1000 == 0 :
        # statistics from most recent training sess
        sample = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
        print(sample)

    X_train = get_next_batch(train_size, current_genre)

    _, D_loss_current = sess.run([D_solver, D_loss], feed_dict={X: X_train, Z: sample_Z(train_size, Z_dim)})
    _, G_loss_current = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(train_size, Z_dim)})
