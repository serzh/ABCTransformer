import sys
import random

import torch
import numpy as np

from tokenizer import vocab_size
from model import AbcTransformer

random.seed(42)
torch.manual_seed(1337)

block_size = 128
batch_size = 128
n_embd = 384
n_heads = 6
n_layers = 6

learning_rate = 3e-4
max_iters = 13000
warmup_iters = 3000

eval_interval = 500
eval_iters = 200
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
autocast_dtype = torch.float16 if device == 'cuda' else torch.bfloat16

print(f'Device: {device}')

with np.load('corpus.npz') as npz:
    train_data = torch.tensor(npz['train_data'])
    val_data = torch.tensor(npz['val_data'])

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ixs = torch.randint(0, len(data) - block_size, (batch_size,))
    xs = []
    ys = []
    for ix in ixs:
        xs.append(data[ix:ix+block_size])
        ys.append(data[ix+1:ix+block_size+1])

    x = torch.stack(xs, dim=0).to(device)
    y = torch.stack(ys, dim=0).to(device)

    return x, y

checkpoint_path = 'checkpoint'

def prepare_model_state(model_state):
    new_state = {}
    prefix = '_orig_mod.'
    for k, v in model_state.items():
        if prefix in k:
            k = k[len(prefix):]
        new_state[k] = v
    return new_state

usage = '''
Usage: python train.py train new
       python train.py train cont <checkpoint-file>
       python train.py sample <checkpoint-file> [input] [seed]
'''

assert len(sys.argv) >= 3, usage

if sys.argv[1] == 'train':
    usage 
    if sys.argv[2] == 'new':
        model_params = dict(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            device=device,
        )
        model = AbcTransformer(**model_params).to(device)
        optimizer_state = None
        min_step = 0

    elif sys.argv[2] == 'cont':
        assert len(sys.argv) >= 4, usage

        checkpoint = torch.load(sys.argv[3])
        model_params = checkpoint['model_params']
        model_state = checkpoint['model']
        optimizer_state = checkpoint['optimizer']
        min_step = checkpoint['step']
        print(model_params)

        model = AbcTransformer(**model_params)
        model.load_state_dict(prepare_model_state(model_state))
        model = model.to(device)



    print(f"Parameters: {sum(p.nelement() for p in model.parameters())/1e6:.1f}M")

    if device == 'cuda':
        model = torch.compile(model)

    def eval():
        out = {}
        model.eval()
        try:
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for i in range(eval_iters):
                    x, y = get_batch(split)
                    logits, loss = model(x, y)
                    losses[i] = loss.item()
                out[split] = losses.mean()
        finally:
            model.train()

        return out
    
    def get_lr(i):
        if i < warmup_iters:
            return learning_rate * i / warmup_iters
        
        return learning_rate


    optimizer = torch.optim.AdamW(lr=learning_rate, params=model.parameters())
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    if device == 'cuda':
        scaler = torch.cuda.amp.grad_scaler.GradScaler()

    for i in range(min_step, max_iters + 1):  

        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = get_batch('train')

        # forward
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            logits, loss = model(x, y)

            if i % eval_interval == 0:
                losses = eval()
                print(f"it {i:8d}: lr {lr:.2E}, mini-batch loss: {loss.item():.4f}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


        if device == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        break

    checkpoint = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        model_params=model_params,
        step=i,
    )
    checkpoint_file = checkpoint_path + str(i) + '.pt'
    torch.save(checkpoint, checkpoint_file)

    print(f"Saved: {checkpoint_file}")

    print("Generated output:")
    with torch.autocast(device, dtype=autocast_dtype):
        print(model.generate_abc(max_tokens=600))


elif sys.argv[1] == 'sample':
    assert len(sys.argv) >= 3, usage

    checkpoint = torch.load(sys.argv[2])
    model_params = checkpoint['model_params']
    model_state = checkpoint['model']
    min_step = checkpoint['step']

    model = AbcTransformer(**model_params)
    model.load_state_dict(prepare_model_state(model_state))
    model = model.to(device)

    if len(sys.argv) >= 4:
        inp = sys.argv[3]
    else:
        inp = None

    if len(sys.argv) >= 5:
        torch.manual_seed(int(sys.argv[4]))

    with torch.autocast(device, dtype=autocast_dtype):
        print(model.generate_abc(inp, 600))

