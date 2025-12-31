# GPT-like Transformer (small educational implementation)

This repository contains a compact, educational implementation of a GPT-style Transformer language model built with PyTorch. It is intended for learning and experimentation rather than production use: the code demonstrates core components such as attention layers, transformer blocks, token embeddings, and a minimal inference loop.

**Goals:**
- Provide a readable, well-structured implementation of a GPT-like model.
- Make it easy to explore attention, normalization, feed-forward layers, and model sizing.
- Offer simple scripts to run quick inference and small tests locally.

**Key characteristics:**
- Implemented with PyTorch (`torch`) and uses `tiktoken` for tokenization in the inference example.
- Focuses on clarity and didactic value over performance and optimizations.

**Repository structure**
- [attention.py](attention.py): implementations of attention variants (SimpleAttention, SelfAttention, CausalAttention, MultiHeadAttention, etc.).
- [transformer.py](transformer.py): `TransformerBlock`, `LayerNorm`, `FeedForward`, and GELU activation used to build the transformer stacks.
- [gpt.py](gpt.py): `GPTModel` wrapper that composes token/position embeddings, transformer blocks, and the output head. Also includes helper methods for printing model summary and parameter counts.
- [config.py](config.py): pre-defined configuration dictionaries (`GPT_CONFIG_SMALL`, `GPT_CONFIG_MEDIUM`, `GPT_CONFIG_LARGE`) with model hyperparameters.
- [inference.py](inference.py): small inference/demo script that tokenizes input with `tiktoken`, runs a forward pass, and decodes generated tokens.
- [test.py](test.py): small utilities and examples for testing attention, layer norm, GELU, and plotting activation functions.
- [playground/](playground): experimental scripts and notebook-style helpers for local experiments and data loading.
- [training-data/](training-data): placeholder folder for datasets and notes (e.g., `the-verdict.txt`).

Getting started
1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
# or .venv\Scripts\activate      # cmd.exe
```

2. Install dependencies (basic):

```powershell
pip install torch tiktoken matplotlib
```

3. Quick inference demo (from repository root):

```powershell
python inference.py
```

Notes on usage and development
- The model is intentionally small by default (`GPT_CONFIG_SMALL`) so you can instantiate, inspect, and run inference without large GPU resources. Feel free to modify `config.py` to experiment with different sizes.
- `gpt.py` expects a configuration dict with keys shown in `config.py`. The forward API takes an input tensor of token indices shaped `(batch_size, seq_len)` and returns logits of shape `(batch_size, seq_len, vocab_size)`.
- The `inference.py` script demonstrates a simple greedy generation loop. Replace the tokenizer or sampling strategy if you want different decoding behavior (top-k, nucleus sampling, temperature scaling).

Testing & experiments
- Run `python test.py` to execute small unit-style snippets that verify attention and normalization behavior and plot activation functions.
- Use the `playground/` folder for prototypes; these scripts are not guaranteed production-ready and may be experimental.

Contributing
- This repository is arranged for educational exploration. Contributions that improve clarity, add tests, or document experiments are welcome. Open an issue or submit a PR.

License
- No explicit license is included in this repository. If you plan to reuse or redistribute the code, add an appropriate `LICENSE` file.

If you'd like, I can also add a `requirements.txt`, improve the inference examples to include sampling, or add a short tutorial notebook—tell me which you'd prefer.
