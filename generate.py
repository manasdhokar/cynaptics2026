import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import re

block_size = 512
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
endw = '</w>'

print("Loading tokenizer...")
with open('tokenizer_data.json', 'r') as f:
    data = json.load(f)

chars = data['vocab']
merges = {(pair[0], pair[1]): pair[0] + pair[1] for pair in data['merges']}
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    words = re.findall(r"\w+|[^\w\s]|\s+", text)
    tokens = []
    for w in words:
        symbols = list(w) + [endw]
        for (first, second) in merges.keys():
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == first and symbols[i+1] == second:
                    merged = first + second
                    symbols = symbols[:i] + [merged] + symbols[i+2:]
                else:
                    i += 1
        tokens.extend([stoi[s] for s in symbols if s in stoi])
    return tokens

def decode(tokens):
    words = [itos[t] for t in tokens]
    text = ''.join(words)
    text = text.replace(endw, '')
    return text

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * ((n_embd//n_head)**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden = int(8 * n_embd // 3)
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(n_embd, hidden, bias=False)
        self.projection = nn.Linear(hidden, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.projection(F.silu(self.w1(x)) * self.w2(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for i in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:,[-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    print(f"Loading model on {device}...")

    model = BigramLanguageModel(vocab_size)

    model.load_state_dict(torch.load('model.pt', map_location=device, weights_only=True))
    model.to(device)

    model.eval()

    start_prompt = input('What do you want your story to start with? ')
    print(f"\nPrompt: '{start_prompt}'")
    print("Generating...")

    encoded_prompt = encode(start_prompt)
    context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=500, temperature=0.5, top_k=50)

    generated_indices = generated_tokens[0].tolist()
    final_text = decode(generated_indices)

    print("\n--- Output ---\n")
    print(final_text)
