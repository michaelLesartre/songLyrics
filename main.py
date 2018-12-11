from genre import genre
import pandas as pd
import sys
import argparse
import os
lyrics = pd.read_csv('lyrics.csv', engine='python', dtype={'lyrics': str})
genres = {}

for name in lyrics.genre.unique():
    if name not in ['Not Available', 'Other']:
        genres[name] = genre(lyrics[lyrics['genre'] == name]['lyrics'], name)

parser = argparse.ArgumentParser()
parser.add_argument('action', action='store')
parser.add_argument('genre', action='store')
parser.add_argument('--seq', action='store', type=int, default=25, dest='seq_length')
parser.add_argument('--wvdim', action='store', type=int, default=100, dest='wv_length')
parser.add_argument('--epochs', action='store', type=int, default=20, dest='epochs')
args = parser.parse_args()

if args.action == 'train':
    genres[args.genre].prepare_model()
    print("model is prepared, starting training")
    genres[args.genre].train_model(epochs=args.epochs)

elif args.action == 'generate':
    genres[args.genre].generate_results(args.seq_length)
