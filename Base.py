# ==============================
# nanoGPT (tiny Shakespeare) — Fully Commented "Base" Version
# Goal: keep the original behavior unchanged, only add clear, didactic comments.
# Style: short, precise, consistent; document shapes & reasons; avoid jargon where possible.
# ==============================

import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------
# Hyperparameters (Karpathy-style tiny config)
# ------------------------------
batch_size   = 16   # how many independent sequences we process in parallel (B)
block_size   = 32   # maximum context length the model can see (T)
max_iters    = 5000 # total training steps
eval_interval= 100  # evaluate every this many steps
learning_rate= 1e-3 # AdamW learning rate (slightly higher for tiny models)
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters   = 200  # number of mini-batches to average for eval loss
n_embd       = 64   # token embedding size (also the model width C)
n_head       = 4    # number of attention heads per block
n_layer      = 4    # number of Transformer blocks
dropout      = 0.0  # dropout disabled for simplicity/fast convergence here

torch.manual_seed(1337)  # reproducibility of sampling, init, batching

# ------------------------------
# Load plain-text corpus (tiny Shakespeare)
# ------------------------------
# Expect an 'input.txt' in the current dir. This is a single long string.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# ------------------------------
# Character-level vocabulary
# ------------------------------
# Build a set of unique characters, then sort for a stable index order.
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Two-way mapping: character <-> integer id
stoi   = {ch: i for i, ch in enumerate(chars)}
itos   = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]            # string -> list[int]
decode = lambda l: ''.join([itos[i] for i in l])   # list[int] -> string

# ------------------------------
# Train/validation split on the tokenized stream
# ------------------------------
# We tokenize once, then split the long 1-D stream into train/val segments.
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))   # 90% for training
train_data = data[:n]
val_data   = data[n:]

# ------------------------------
# Mini-batch loader: returns (x, y) pairs
# ------------------------------
# x: (B, T) current tokens
# y: (B, T) target tokens = tokens shifted by +1 (next-char prediction)
def get_batch(split: str):
    # Choose source stream
    src = train_data if split == 'train' else val_data
    # Random start indices so that [i, i+T) and [i+1, i+T+1) are valid slices
    ix = torch.randint(len(src) - block_size, (batch_size,))
    # Stack contiguous windows into a batch
    x = torch.stack([src[i : i + block_size]       for i in ix])  # (B, T)
    y = torch.stack([src[i + 1 : i + block_size+1] for i in ix])  # (B, T)
    # Move to target device
    x, y = x.to(device), y.to(device)
    return x, y

# ------------------------------
# Loss estimator (no grad): returns mean train/val loss over several mini-batches
# ------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # turn off dropout, use running stats if any
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)  # forward computes loss when targets provided
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # back to training mode
    return out

# ==============================
# Transformer components
# ==============================

class Head(nn.Module):
    """
    Single self-attention head.
    Input:  x of shape (B, T, C) where C == n_embd
    Output: out of shape (B, T, head_size)
    """
    def __init__(self, head_size: int):
        super().__init__()
        # Linear projections (no bias) to key, query, value spaces, each (C -> head_size)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask (lower triangular): (T, T) with ones on/below diagonal
        # register_buffer => not a parameter; moves with the module across devices.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape  # C is n_embd
        # Project inputs to key/query/value for each time step
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Scaled dot-product attention scores:
        # q @ k^T over the last dim => (B, T, T). Scale by sqrt(head_size) to stabilize softmax.
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)  # (B, T, T)

        # Apply causal mask so position t can only attend to <= t (no future leakage)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Softmax over the last dim to get attention weights per query token
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Weighted sum of values
        v   = self.value(x)           # (B, T, head_size)
        out = wei @ v                 # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention: run several Heads in parallel and concat.
    Input:  (B, T, C)
    Output: (B, T, C) after a final projection back to C.
    """
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)  # combine concatenated heads back to model dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # Concatenate along channel dim: (B, T, num_heads*head_size) == (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))  # final linear mixing + dropout
        return out

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network applied at each time step independently.
    Shape-preserving: (B, T, C) -> (B, T, C).
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # hidden expansion (Transformer paper commonly uses 4x)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # project back to model width
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class Block(nn.Module):
    """
    Transformer block: LayerNorm -> Self-Attention -> residual,
                        LayerNorm -> FeedForward    -> residual.
    All shape-preserving on (B, T, C).
    """
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head  # must divide evenly
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor):
        # Pre-norm architecture: LN before each sub-layer; add residual after each.
        x = x + self.sa(self.ln1(x))  # (B, T, C)
        x = x + self.ffwd(self.ln2(x))# (B, T, C)
        return x

# ==============================
# Language Model (character-level)
# ==============================
class TransformerLM(nn.Module):
    """
    A minimal GPT-style decoder-only Transformer for next-character prediction.

    Inputs:
      - idx: LongTensor of shape (B, T) with token ids
      - targets: optional LongTensor (B, T) for loss computation

    Returns:
      - logits: (B, T, vocab_size) unnormalized scores per token position
      - loss:   scalar cross-entropy if targets is not None, else None
    """
    def __init__(self):
        super().__init__()
        # Token and positional embeddings produce (B, T, C) that are summed
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd) # maps token id -> vector (C)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # maps position 0..T-1 -> vector (C)

        # Stack of Transformer blocks (depth = n_layer)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)              # final LayerNorm
        self.lm_head= nn.Linear(n_embd, vocab_size)     # project to vocabulary logits

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape  # (batch, time)

        # 1) Embed tokens and positions, then sum to get contextualized inputs
        tok_emb = self.token_embedding_table(idx)                          # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb                                              # (B, T, C)

        # 2) Transformer body
        x = self.blocks(x)                                                 # (B, T, C)
        x = self.ln_f(x)                                                   # (B, T, C)

        # 3) Project to logits over the vocabulary for each position
        logits = self.lm_head(x)                                           # (B, T, vocab_size)

        # 4) Optional loss: flatten batch+time to one dimension for cross-entropy
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)        # (B*T, vocab_size)
            targets = targets.view(B * T)          # (B*T,)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Autoregressive sampling:
          - idx: (B, T) current context token ids
          - Iteratively append 1 token at a time, up to `max_new_tokens`
          - Always crop context to the last `block_size` tokens (causal model limit)
        Returns:
          - idx: (B, T + max_new_tokens) with newly sampled tokens appended
        """
        for _ in range(max_new_tokens):
            # 1) Crop to the model's max context length so shapes stay <= (B, block_size)
            idx_cond = idx[:, -block_size:]  # (B, min(T, block_size))

            # 2) Forward pass to obtain next-token distribution for the last position
            logits, _ = self(idx_cond)       # logits: (B, t, vocab_size)
            logits = logits[:, -1, :]        # only need the distribution at the final step (B, vocab_size)

            # 3) Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # 4) Sample one token id per batch from the categorical distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 5) Append to the running sequence (grows time dimension by 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

# ------------------------------
# Instantiate model & optimizer
# ------------------------------
model = TransformerLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ------------------------------
# Training loop
# ------------------------------
for iter in range(max_iters):

    # Periodically estimate mean train/val loss (helps sanity-check overfitting)
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 1) Sample a fresh mini-batch
    xb, yb = get_batch('train')          # xb, yb: (B, T)

    # 2) Forward pass + loss
    logits, loss = model(xb, yb)

    # 3) Backprop + optimizer update
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ------------------------------
# Text generation demo
# ------------------------------
# Start from a single "zero" token (matches whatever char id 0 is); shape (1, 1).
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Sample 500 new tokens autoregressively, then decode ids back to text.
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))
