import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import encode, decode

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embd, dropout):
        super().__init__()
        self.dropout = dropout
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.QKV = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)


    def forward(self, x):
        B, T, C = x.shape
        qkv = self.QKV(x)
        q,k,v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if not self.training else 0.0)
        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd, bias=False),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_heads, n_embd, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_heads, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    
class AbcTransformer(nn.Module):
    def __init__(self, *, vocab_size, n_embd, block_size, n_heads, n_layers, dropout, device):
        super().__init__()
        self.device = device
        self.block_size = block_size

        self.embeddings = nn.Embedding(vocab_size, n_embd)
        self.pos_embeddings = nn.Embedding(block_size, n_embd)
        self.sa_blocks= nn.Sequential(*[Block(n_heads, n_embd, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.ln_head = nn.Linear(n_embd, vocab_size, bias=False)
        
    def forward(self, x, y=None):
        word_embeddings = self.embeddings(x)
        pos_embeddings = self.pos_embeddings(torch.arange(x.shape[1], device=self.device))
        x = word_embeddings + pos_embeddings
        x = self.sa_blocks(x)
        x = self.ln(x)
        x = self.ln_head(x)
        logits = x
        if y is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        return logits, loss


    def generate(self, idx, max_tokens):
        self.eval()
        try:
            for _ in range(max_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_n = torch.multinomial(probs, num_samples=1)
                if idx_n == 0:
                    break
                idx = torch.cat((idx, idx_n), dim=-1)
            return idx
        finally:
            self.train()

    def generate_abc(self, inp = "", max_tokens=200):
        inp = torch.tensor([[0] + encode(inp)], dtype=torch.long) if inp else torch.zeros((1, 1), dtype=torch.long) 
        return 'X:1\n' + decode(self.generate(inp.to(self.device), max_tokens)[0].tolist()[:]) 
