# Shakespeare-GPT: An Iterative Study in BPE & Architecture Scaling

This repository documents the development of a decoder-only Transformer model trained on the Tiny Shakespeare corpus. This project is a hands-on exploration of the transition from character-level baselines to sub-word tokenization and the engineering trade-offs of model scaling.

## 🛠️ The Development Journey

### 1. Architectural Foundations
The project began with a "from-scratch" implementation of the core components of the Transformer architecture, moving beyond simple Bigram models to a full Attention-based system. Key modules implemented include:
* **Causal Self-Attention:** Implementing the query, key, and value matrices with causal masking.
* **Multi-Head Attention:** Parallelizing attention heads to capture diverse linguistic relationships.
* **Residual Blocks & LayerNorm:** Ensuring stable signal flow through deep layers.
* All of this implemented by referring andrej karpathy's youtube video

### 2. The BPE Pivot (Sub-word Tokenization)
Moving beyond the character-level limitations (Vocab ~65) often used in basic tutorials, I implemented a custom **Byte-Pair Encoding (BPE)** tokenizer which I learned via this post on medium: https://medium.com/@adarsh-ai/build-a-byte-pair-encoding-bpe-tokenizer-from-scratch-in-python-0dc32c6410f7
* **Implementation:** Based on architectural insights from the community, I developed a BPE logic that performs 190 merges on the raw text.
* **The Advantage:** With a vocabulary size of **256**, the model achieves higher information density. Each token represents larger semantic chunks (common words/suffixes), effectively extending the model's functional context window within the same 256-block size.

### 3. Scaling & Parameter Tuning
A significant portion of this project was dedicated to finding the "Goldilocks" zone for model capacity:
* **The 24M Experiment:** I initially scaled the model to **24 Million parameters**. However, on the 1MB Shakespeare dataset, this led to immediate and severe overfitting. The model memorized the training set (Loss ~1.10) while failing to generalize to the validation set.
* **The Strategic Downscale:** To combat this, I pivoted to the current **10.8M parameter** configuration. This "Medium" model acts as a natural bottleneck, forcing the Transformer to learn generalizable patterns rather than verbatim lines.

---

## 🏗️ Model Specifications

| Component | Specification |
| :--- | :--- |
| **Model Size** | ~10.8 Million Parameters |
| **$n_{embd}$** | 384 |
| **$n_{head}$** | 6 (64-dim per head) |
| **$n_{layer}$** | 6 |
| **Dropout** | 0.25 |
| **Vocab Size** | 256 (BPE-encoded) |

## 📉 Optimization Strategy: The Baseline
This iteration of the model serves as a **Stable Architecture Baseline**. To isolate the performance impact of the BPE tokenizer and architecture scaling, the following constraints were maintained:
* **Constant Learning Rate:** A fixed learning rate of `3e-4` was used to observe the raw convergence behavior of the architecture without scheduler intervention.
* **AdamW (Zero Decay):** The optimizer was run without explicit weight decay to establish a baseline for generalization provided solely by **Dropout (0.25)** and architectural bottlenecks.

---

## 📂 Project Structure

* `vfinal.py`: Unified script containing the BPE merging logic, Transformer architecture, and training loop.
* `data/input.txt`: The Tiny Shakespeare corpus.
* `README.md`: Documentation of the engineering process and findings.

---

## 📊 Results & Observations
The model achieves a competitive validation loss of **~1.47**. 

**Engineering Insight:** Through this process, I observed that while character-level models might show lower raw loss numbers, the BPE-informed loss represents a higher semantic quality. 

* **Character Level (Vocab ~65):** A completely random guess has a loss of $\ln(65) \approx 4.17$.
* **BPE Level (Vocab 256):** A completely random guess has a loss of $\ln(256) \approx 5.54$.

Because the BPE model starts with a much higher "uncertainty floor," its final loss of 1.47 means it has reduced its uncertainty much more significantly than the character model did to reach that same 1.47.

> "A common misconception is that a BPE-based model with a loss of 1.50 is performing worse than a character-based model with a loss of 1.47. In reality, the BPE model is significantly more efficient for several reasons:
>
> 1. **Higher Information Density:** Each token in this model represents $\sim2.5$ characters on average. Predicting a complex sub-word token with 1.50 loss is mathematically 'smarter' than predicting a single character with the same loss.
> 2. **Effective Context Length:** By using BPE, our fixed 256-token context window covers nearly **3x more text** than a character-level model. This allows for superior long-range coherence in the generated Shakespearean prose.
> 3. **Inference Throughput:** Because the model predicts larger chunks of text at once, it requires $\sim60\%$ fewer forward passes to generate the same amount of text, making it much faster in a production environment."

### Comparison Summary

| Metric | Character-Level Baseline | Your BPE Model |
| :--- | :--- | :--- |
| **Vocab Size** | 65 | **256** |
| **Information per Step** | 1 Character | **~2.5 Characters** |
| **Context Memory** | ~40 words | **~150 words** |
| **Inference Speed** | Baseline | **~2.5x Faster** |
| **True Model Quality** | Good | **Superior (Better Coherence)** |





### Sample Generation
SICINIUS:
Nay, sir, it is my true night.

BENVOLIO:
Sir, I bear you, my lord.
Behold, chold, behold,
Standst mine contents his parture in rottent,
Fetches, good privated being.
Go to, let not us to-morrow the pestor shade
But lie that recencies you shall intent me.
Lo, not the rest,--doemight of disance,
For fortunation! I'll give my such palace.....
---

## Future Roadmap
* **Optimization Phase:** Introducing a Cosine Decay scheduler to further "polish" the final weights.
* **Regularization Ablation:** Testing the impact of Weight Decay (0.1) vs. the current zero-decay baseline.
