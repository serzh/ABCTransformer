import os
import pickle
import random

import numpy as np

random.seed(42)

skip_fields = ['%', 'n:', 'x:', 't:', 'c:', 'z:', 'w:', 'b:', 'd:', 'a:', 'h:', 'i:', 's:', 'r:', 'o:', 'f:']
songs = []
dump_dir = 'data/'
for fn in sorted(os.listdir(dump_dir)):
    if fn.endswith('.abc'):
        with open(dump_dir + fn, 'rt', encoding='latin-1') as f:
            for song in f.read().split('\n\n\n'):
                song_lines = []
                for song_line in song.split('\n'):
                    if not song_line or song_line.isspace():
                        continue
                    if any(song_line.lower().startswith(sf) for sf in skip_fields):
                        continue
                        
                    song_lines.append(song_line)
                if song_lines:
                    songs.append('\n'.join(song_lines))

print(f"Total: {len(songs)} songs")
vocab = sorted(set(''.join(songs)))

stoi = {s:i+1 for i,s in enumerate(vocab)}

itos = {i:s for s,i in stoi.items()}
itos[0] = ''

with open('itos.pkl', 'wb') as f:
    pickle.dump(itos, f)

from tokenizer import encode

data = [0]
random.shuffle(songs)
for song in songs:
    data += encode(song) + [0]

n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train tokens: {len(train_data)/1e6:.1f}M")
print(f"Valdation tokens: {len(val_data)/1e6:.1f}M")
print(f"Total tokens: {(len(train_data) + len(val_data))/1e6:.1f}")

np.savez_compressed('corpus.npz', train_data=np.array(train_data, dtype=np.int64), val_data=np.array(val_data, dtype=np.int64))
print("Saved dataset into corpus.npz")