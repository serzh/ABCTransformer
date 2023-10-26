import pickle

with open('itos.pkl', 'rb') as f:
    itos = pickle.load(f)

stoi = {s:i for i,s in itos.items()}

vocab_size = len(itos)

encode = lambda s: [stoi[c] for c in s]
decode = lambda a: ''.join([itos[i] for i in a])