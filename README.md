# TinyStories GPT — 11.6M Parameter Transformer From Scratch

A GPT-style language model built **from scratch in Python using PyTorch**, trained on a subset of the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

The final model has **11.6M parameters** and is capable of generating coherent, grammatically correct children's stories.

The goal of this project was not simply to train a language model, but to understand **how modern LLMs work internally** by implementing the core components myself rather than relying on existing tokenizer or model libraries.

---

## 🚀 Highlights

- **11.6M parameter GPT-style Transformer**
- Transformer architecture implemented **from scratch in PyTorch**
- **BPE tokenizer implemented from scratch**
- Multi-head causal self-attention
- **SwiGLU** feed-forward network
- Top-K sampling
- Temperature scaling
- Cosine learning-rate scheduling
- Custom dataset preprocessing pipeline
- **Memory-mapped binary token storage** for large-scale dataset processing
- **Multiprocessing** for faster preprocessing
- Training & validation loss tracking
- GPU training using **Modal**
- Trained on **25% of the TinyStories dataset**
- Generates coherent children's stories from scratch

---

# 🧠 Architecture

The model follows a decoder-only Transformer architecture similar to the core architecture used by GPT-style language models.

```text
Input Text
    ↓
BPE Tokenizer
    ↓
Token IDs
    ↓
Token Embeddings + Positional Embeddings
    ↓
┌───────────────────────────────┐
│     Transformer Block × 6     │
│                               │
│  LayerNorm                    │
│      ↓                        │
│  Multi-Head Self-Attention    │
│      ↓                        │
│  Residual Connection          │
│      ↓                        │
│  LayerNorm                    │
│      ↓                        │
│  SwiGLU Feed-Forward Network  │
│      ↓                        │
│  Residual Connection          │
└───────────────────────────────┘
    ↓
Language Model Head
    ↓
Logits
    ↓
Temperature + Top-K Sampling
    ↓
Generated Text
```

## Model Configuration

| Parameter | Value |
|---|---:|
| Parameters | **11.6M** |
| Embedding dimension | 384 |
| Attention heads | 6 |
| Transformer layers | 6 |
| Context length | 512 |
| Dropout | 0.2 |
| Activation | **SwiGLU** |
| Tokenizer | Custom BPE |
| Architecture | Decoder-only Transformer |

---

# 🔤 BPE Tokenizer — Built From Scratch

One of the main goals of this project was to avoid treating tokenization as a black box.

I implemented a **Byte Pair Encoding (BPE) tokenizer from scratch** based on the approach described [here](https://medium.com/@adarshpritam/build-a-byte-pair-encoding-bpe-tokenizer-from-scratch-in-python-0dc32c6410f7).

The tokenizer:

1. Processes the training corpus
2. Finds frequently occurring byte/subword pairs
3. Learns merge rules
4. Converts text into token IDs
5. Converts generated token IDs back into readable text

This gave me a much better understanding of how raw text eventually becomes the numerical representation consumed by a Transformer.

---

# ⚡ SwiGLU

I experimented with replacing the standard ReLU activation in the Transformer feed-forward network with **SwiGLU**.

SwiGLU is widely used in modern Transformer architectures and introduces a gated mechanism into the feed-forward layer.

Reference:  
https://medium.com/@s_boudefel/exploring-swiglu-the-activation-function-powering-modern-llms-9697f88221e7

This was one of several architectural changes I experimented with while improving generation quality.

---

# 🎲 Text Generation

The model supports two sampling techniques.

### Temperature Scaling

Temperature controls how concentrated or diverse the probability distribution is during generation.

- Lower temperature → more deterministic
- Higher temperature → more diverse/random

### Top-K Sampling

Instead of sampling from the entire vocabulary, the model restricts the candidate tokens to the **K most probable tokens**.

Combining Top-K sampling with temperature scaling gives significantly more control over the generated text.

---

# 📉 Cosine Learning-Rate Scheduling

I implemented cosine learning-rate decay rather than keeping the learning rate constant throughout training.

The learning rate follows a cosine schedule between:

```text
Initial LR = 6e-4
Minimum LR = 6e-5
```

This gradually reduces the learning rate as training progresses, allowing larger updates earlier in training and smaller updates later.

---

# 🛠️ The Journey

This project started much smaller.

## Step 1 — Tiny Shakespeare

I initially followed Andrej Karpathy's **Neural Networks: Zero to Hero** series:

https://lnkd.in/dwQGfujw

The first version was a small GPT-style model trained on the **Tiny Shakespeare dataset (~1 MB)**.

At this stage, the primary objective was simply understanding the fundamentals:

- Token embeddings
- Self-attention
- Multi-head attention
- Transformer blocks
- Residual connections
- Layer normalization
- Autoregressive generation
- Next-token prediction

This gave me the foundation to start modifying the architecture myself.

---

## Step 2 — Building My Own Tokenizer

The next step was removing another abstraction.

Instead of using an existing tokenizer, I implemented **BPE tokenization from scratch**.

This forced me to understand the entire pipeline:

```text
Raw Text
   ↓
Tokenization
   ↓
Token IDs
   ↓
Embedding
   ↓
Transformer
   ↓
Logits
   ↓
Sampling
   ↓
Generated Text
```

This was an important transition from simply reproducing a GPT implementation to understanding the complete language-model pipeline.

---

## Step 3 — Improving Text Generation

I then implemented:

- **Top-K sampling**
- **Temperature scaling**

These allowed me to control the trade-off between deterministic and diverse generation.

At this point, the model was already producing noticeably better outputs.

---

## Step 4 — Experimenting With the Architecture

I replaced the standard **ReLU FFN activation with SwiGLU**.

This was another opportunity to experiment with the architecture rather than treating the Transformer as a fixed implementation.

Reference:

https://lnkd.in/dBsY5uu9

---

## Step 5 — The Overfitting Problem

After these improvements, I ran into a major problem:

**the model was overfitting.**

The initial dataset was simply too small relative to the model.

Instead of immediately changing the model architecture, I looked at the relationship between **model size and training data**.

I therefore scaled the training data to approximately **25% of the TinyStories dataset**, giving the model substantially more training tokens.

This was also where the project changed from primarily an architecture experiment into a **data and systems engineering problem**.

---

# 💾 Scaling the Dataset

Processing a much larger dataset introduced a new bottleneck.

Loading and processing everything directly in RAM was inefficient.

So I changed the preprocessing pipeline to:

```text
TinyStories
     ↓
Batch Processing
     ↓
BPE Tokenization
     ↓
Token IDs
     ↓
Binary Files
     ↓
Memory Mapping
     ↓
Training
```

Instead of keeping the entire tokenized dataset in memory, I stored the token IDs in binary files and used **memory mapping** to access the data efficiently.

This significantly reduced RAM requirements and made it practical to work with a much larger corpus.

---

# 🚀 Multiprocessing

The next bottleneck was CPU preprocessing.

Tokenizing a large dataset sequentially using a single Python process was slow.

I therefore implemented **multiprocessing** so that multiple CPU cores could process batches concurrently.

Conceptually:

```text
                Dataset
                   │
        ┌──────────┼──────────┐
        ↓          ↓          ↓
     Worker 1   Worker 2   Worker 3   ...
        │          │          │
        └──────────┼──────────┘
                   ↓
            Binary Token Data
```

This made dataset preprocessing substantially faster than processing everything through a single loop.

---

# 📈 Training

The final model was trained using GPU infrastructure through **Modal**.

During training, I tracked both:

- Training loss
- Validation loss

This allowed me to monitor whether the model was actually learning the underlying distribution or simply memorizing the training data.

The final training curves show the losses converging as training progresses.

---

# 🧪 What I Learned

The most valuable part of the project was seeing how seemingly small changes affected the final model.

I learned that building an LLM isn't just about implementing attention.

The complete system involves:

```text
Data
 ↓
Tokenization
 ↓
Dataset Engineering
 ↓
Model Architecture
 ↓
Optimization
 ↓
Training Infrastructure
 ↓
Sampling
 ↓
Evaluation
```

A bottleneck at any one of these stages can limit the final model.

For example:

- A better architecture doesn't help much if the dataset is too small.
- A large dataset becomes difficult to work with without efficient preprocessing.
- Efficient preprocessing doesn't matter if the training setup is poorly configured.
- A trained model still needs appropriate sampling to generate useful text.

---

# 📊 Results

The final **11.6M parameter model** is capable of generating coherent children's stories with generally correct grammar and locally consistent narratives.

Example generations and training curves are shown below.

![Training Loss](https://github.com/user-attachments/assets/63420fad-fedb-4a7d-bc12-2429b67e0a3b)

![Model Results](https://github.com/user-attachments/assets/25827225-5616-4fc4-b978-f273d7881185)

The model is small compared with modern LLMs, but building it end-to-end made it possible to understand the fundamental mechanisms behind them.

---

# 🔬 References

### Andrej Karpathy — Neural Networks: Zero to Hero

https://lnkd.in/dwQGfujw

### TinyStories Dataset

https://huggingface.co/datasets/roneneldan/TinyStories

### BPE Tokenizer

https://medium.com/@adarshpritam/build-a-byte-pair-encoding-bpe-tokenizer-from-scratch-in-python-0dc32c6410f7

### SwiGLU

https://medium.com/@s_boudefel/exploring-swiglu-the-activation-function-powering-modern-llms-9697f88221e7

---

# 🎯 Final Takeaway

What started as a **1 MB Tiny Shakespeare experiment** gradually became a complete miniature language-model training pipeline.

I went from:

**following a GPT implementation → building my own tokenizer → experimenting with the architecture → solving overfitting → scaling the dataset → optimizing preprocessing → training on GPU infrastructure → generating coherent stories.**

The most rewarding part was watching the model improve after every incremental change and understanding **why** each change affected the final behavior.

This project gave me a much deeper understanding of what actually happens underneath modern LLM APIs and motivated me to continue applying machine learning to different domains.
