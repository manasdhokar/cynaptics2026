import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import json
import modal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. Modal Setup
app = modal.App("tinystories-trainer")
image = modal.Image.debian_slim().pip_install("torch", "numpy", "matplotlib")
volume = modal.Volume.from_name("tinystories-volume", create_if_missing=True)
DATA_DIR = "/data"

# hyperparameters
batch_size = 64 
block_size = 512 
max_iters = 15200
eval_interval = 100
learning_rate = 6e-4 
min_learning_rate = 6e-5 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 
n_head = 6
n_layer = 6
dropout = 0.2 

torch.manual_seed(1337)

# Helper functions adapted to accept train_data and val_data explicitly
def get_batch(split, train_data, val_data):  
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()     
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval() # disables dropout
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()  
    model.train() # enables dropout
    return out

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
    def __init__(self, vocab_size): # <-- Now accepts vocab_size dynamically
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

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature, top_k):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) 
                logits[logits < v[:,[-1]]] = float('-inf') 

            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx


def get_lr(iter):
    A = (learning_rate - min_learning_rate) * 0.5
    c = iter / max_iters
    lr = min_learning_rate + A * (1 + math.cos(math.pi * c))
    return lr


# 2. Remote execution function with GPU request and extended timeout
@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L40S", timeout=86400)
def train_model():
    print(f"Running on device: {device}")

    # Load Tokenizer Data from the Modal Volume
    tokenizer_path = f"{DATA_DIR}/tokenizer_data.json"
    print(f"Loading vocabulary from {tokenizer_path}...")
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    chars = tokenizer_data['vocab']
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")

    # Load Training Data from the Modal Volume
    print("Loading binary token data...")
    train_data = np.memmap(f'{DATA_DIR}/tinystories_train_tokens.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap(f'{DATA_DIR}/tinystories_validation_tokens.bin', dtype=np.uint16, mode='r')

    # Initialize model with the dynamic vocab_size
    model = BigramLanguageModel(vocab_size=vocab_size)
    m = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    history = {'iter': [], 'step_iter':[], 'train_loss': [], 'val_loss': [], 'lr': []}

    print("Starting training loop...")
    for iter in range(max_iters):  
        lr = get_lr(iter)
        for p in optimizer.param_groups:
            p['lr'] = lr
        
        history['step_iter'].append(iter)
        history['lr'].append(lr)

   
        if iter % eval_interval == 0:
            losses = estimate_loss(m, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
            history['iter'].append(iter)
            history['train_loss'].append(float(losses['train']))
            history['val_loss'].append(float(losses['val']))
            

        xb, yb = get_batch('train', train_data, val_data)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True) 
        loss.backward() 
        optimizer.step() 
    
    # Save the model directly to the Volume
    save_path = f'{DATA_DIR}/model.pt'
    torch.save(model.state_dict(), save_path) 

    # --- NEW: Generate and save the plot ---
    print("Generating training graphs...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Train vs Validation Loss
    zoom_start = min(3, len(history['iter']) - 1)

    axs[0].plot(history['iter'][zoom_start:], history['train_loss'][zoom_start:], label='Train Loss', color='blue')
    axs[0].plot(history['iter'][zoom_start:], history['val_loss'][zoom_start:], label='Val Loss', color='green', linestyle= '--')
    axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training & Validation Loss')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Learning Rate Decay
    axs[1].plot(history['step_iter'], history['lr'], label='Learning Rate', color='purple')
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Learning Rate')
    axs[1].set_title('Cosine Learning Rate Decay')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Save the plot to the volume
    plt.tight_layout()
    plot_path = f'{DATA_DIR}/training_metrics.png'
    plt.savefig(plot_path)
    
    # Also save the raw history data as a JSON file just in case you want exact numbers later
    with open(f'{DATA_DIR}/training_history.json', 'w') as f:
        json.dump(history, f)
    
    # Commit changes so the saved model persists in the cloud
    volume.commit()
    print(f"Parameters successfully saved to {save_path} and volume committed.")


# 3. Local entrypoint
@app.local_entrypoint()
def main():
    print("Kicking off remote GPU training job on Modal...")
    train_model.remote()
    print("Training finished.")