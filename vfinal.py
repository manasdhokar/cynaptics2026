import torch
import torch.nn as nn
from torch.nn import functional as F
import collections

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd= 384 
n_head=6 
n_layer=6
dropout=0.25
# ------------

torch.manual_seed(1337)

#tokenizer 
from datasets import load_dataset
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read() #converts to string

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))

endw='</w>'
chars.append(endw)

#pre tokenization- split into words, add endw at end of each word
import re
word_splits={}
words = re.findall(r"\w+|[^\w\s]", text)  #splits word/number+ anything that is not words or whitespace

for w in words:
    chars_in_word= list(w) + [endw]

    word_tuple= tuple(chars_in_word)

    if word_tuple not in word_splits:
        word_splits[word_tuple] = 0 #initializes key if it is not alr there

    word_splits[word_tuple]+=1

def get_pair_stats(splits): #pairs adjacent characters andn counts their frequency
    
    pair_counts= collections.defaultdict(int) #collections dictionary will create new key if dne, defaultdict- def val=0
    
    for word_tuple, freq in splits.items():
        symbols= list(word_tuple)

        for i in range(len(symbols)-1):
            pair=(symbols[i], symbols[i+1])
            pair_counts[pair]+= freq
    return pair_counts

def merge_pair(pair_to_merge, splits):
    new_splits= {}
    (first_char, second_char)= pair_to_merge
    merged_token= first_char + second_char

    for word_tuple, freq in splits.items():
        symbols= list(word_tuple)
        new_symbols=[]

        i=0
        while i < len(symbols):
            if i < len(symbols)-1 and symbols[i] == first_char and symbols[i+1] == second_char:
                new_symbols.append(merged_token)
                i+=2
            else:
                new_symbols.append(symbols[i])
                i+=1
        new_splits[tuple(new_symbols)] = freq
    return new_splits

#BPE merging
num_merges= 190
merges = {}
current_splits=word_splits.copy()

for i in range(num_merges):
    
    pair_stats = get_pair_stats(current_splits)

    if not pair_stats:
        break
    
    #finding best pair
    best_pair = max(pair_stats, key=pair_stats.get) #key means what is to be checked
    best_freq = pair_stats[best_pair]

    #merging the best pair into a new token
    current_splits = merge_pair(best_pair, current_splits)
    new_token= best_pair[0] + best_pair[1]

    #storing the merge
    merges[best_pair] = new_token

    
    #updating vocab
    chars.append(new_token)

vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }


def encode(text):
    words = re.findall(r"\w+|[^\w\s]", text)
    tokens = []
    
    for w in words:
        symbols = list(w) + [endw]
        
        # apply merges in same order they were learned
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
    text = text.replace(endw, ' ')  # replace end of word marker with space
    return text


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()     #this entire blkci is to chek how good the model is working at step x wtihout training it. torch.no_grad() disables gradient tracking
def estimate_loss():
    out = {}
    model.eval() #disables doropout
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()  #the loss you see is mean of the number of evaluation iterations not the final loss
    model.train() #enables dropout. ie model shifts to training mode.
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key= nn.Linear(n_embd, head_size, bias= False)
        self.query= nn.Linear(n_embd, head_size, bias= False)
        self.value= nn.Linear(n_embd, head_size, bias= False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout= nn.Dropout(dropout)
    def forward(self, x):
        B,T,C= x.shape
        k= self.key(x)
        q= self.query(x)

        wei= q @ k.transpose(-2,-1)*C**-0.5 #GIVES B,T,T
        wei= wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei= F.softmax(wei, dim=-1) #created the attention scores BTT 
        wei= self.dropout(wei)

        v=self.value(x) #BTC
        out=wei @ v #Shape of out is BTC
        return out

class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout= nn.Dropout(dropout)

    def forward(self, x):
        out=torch.cat([h(x) for h in self.heads], dim=-1)
        out=self.proj(out)
        out=self.dropout(out)
        return out  #all head's outs are are concatenated along channel, so B,T, n_embd*num_heads

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        ) #hands data from prev layer to next layer, reduces statements in the forward funciton
    
    def forward(self, x):
     return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd= FeedForward(n_embd)
        self.ln1= nn.LayerNorm(n_embd)
        self.ln2= nn.LayerNorm(n_embd)

    def forward(self, x):
            x=x+ self.sa(self.ln1(x))
            x=x+ self.ffwd(self.ln2(x))
            return x




class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table=nn.Embedding(block_size, n_embd)
        self.blocks=nn.Sequential( 
        *[Block(n_embd, n_head=n_head) for i in range(n_layer)]
        ) 
        self.ln_f= nn.LayerNorm(n_embd)
        self.lm_head= nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        B,T=idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb= self.token_embedding_table(idx) #(B,T,n_embd)
        pos_emb= self.positional_embedding_table(torch.arange(T, device=device)) #(passing 0,1,..T-1 gets T,emb_d tensor from table)
        x=tok_emb+pos_emb # B,T,n_embd (concept-brodcasting)
        x=self.blocks(x)
        x=self.ln_f(x)
        logits= self.lm_head(x)# (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond= idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) #set to zero
    loss.backward() #fresh gradients calculated for current loss
    optimizer.step() #model parameters updated (embedding table, all the weights, biases etc) not logits as they are computed and are not the learnable parameters.

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2500)[0].tolist()))
