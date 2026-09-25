# TinyStories GPT — 11.6M Parameter Transformer From Scratch

A GPT-style language model built **from scratch in Python using PyTorch**, trained on a subset of the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

The model has **11.6M parameters** and is capable of generating coherent, grammatically correct children's stories.

## Overview

I initially started this project by following Andrej Karpathy's [Neural Networks: Zero to Hero](https://lnkd.in/dwQGfujw) video series, where the initial model was trained on the **Tiny Shakespeare** dataset (~1 MB).

From there, I progressively extended the architecture and training pipeline to build a more capable GPT-style language model.

The main goal was to understand how modern language models work by implementing the core components myself rather than relying on existing tokenizer or model libraries.

## Features

- **11.6M parameter GPT-style Transformer**
- Transformer architecture implemented from scratch in PyTorch
- **BPE tokenizer implemented from scratch**
- Top-K sampling
- Temperature scaling
- Cosine learning-rate scheduling
- **SwiGLU** activation in the Feed-Forward Network
- Custom dataset preprocessing pipeline
- Binary token storage using memory-mapped files
- Multiprocessing for faster dataset preprocessing
- Training and validation loss tracking
- GPU training using Modal

## Architecture

The model uses a decoder-only Transformer architecture consisting of:

- Token embeddings
- Positional embeddings
- Multi-head self-attention
- Feed-forward networks using **SwiGLU**
- Residual connections
- Layer normalization
- Linear language-model head

### Model Configuration

| Parameter | Value |
|---|---:|
| Parameters | **11.6M** |
| Embedding dimension | 384 |
| Attention heads | 6 |
| Transformer layers | 6 |
| Context length | 512 |
| Dropout | 0.2 |
| Vocabulary | Custom BPE tokenizer |

## BPE Tokenizer

I implemented a **Byte Pair Encoding (BPE) tokenizer from scratch** based on the approach described in:

- [Andrej Karpathy — Let's build the GPT Tokenizer](https://lnkd.in/d2ejhM-C)

The tokenizer learns frequent subword merges from the training corpus and converts text into a sequence of token IDs that can be processed by the Transformer.

The tokenizer also supports converting generated token sequences back into readable text.

## Sampling

The model supports two techniques for controlling text generation:

### Temperature Scaling

Temperature controls the randomness of the probability distribution used during generation.

- Lower temperature → more deterministic outputs
- Higher temperature → more diverse outputs

### Top-K Sampling

Instead of sampling from the entire vocabulary, the model restricts sampling to the **K most probable tokens**.

Combining Top-K sampling with temperature scaling gives more control over the generated stories.

## SwiGLU

I replaced the standard ReLU activation in the Transformer Feed-Forward Network with **SwiGLU**.

Reference:

- [SwiGLU — GLU Variants Improve Transformer](https://lnkd.in/dBsY5uu9)

This was one of the architectural changes I experimented with to improve the model's generation quality.

## Learning Rate Scheduling

I implemented **cosine learning-rate decay**, gradually reducing the learning rate during training.

The learning rate follows a cosine schedule between:

```text
Initial LR = 6e-4
Minimum LR = 6e-5
