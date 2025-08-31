import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import prune  # Added: pruning utilities from PyTorch
import math

# =========================
# Tunable knobs (for one-click runs)
# =========================
TARGET_SPARSITY = 0.50     # Target sparsity (0..1). 0.5 or 0.8 are classic demo values.
GLOBAL_PRUNE   = True      # True = global magnitude pruning; False = per-layer pruning
PRUNE_AT_ITER  = 3000      # Apply pruning at this training step (must be < max_iters)
FINETUNE_ITERS = 500       # Fine-tuning steps after pruning to recover accuracy
REPORT_TOPK    = 10        # Show sparsity stats for the top-K most sparse layers
SEED           = 1337

# =========================
# Baseline hyperparameters (kept same or lightly tweaked)
# =========================
# Karpathy's hyperparameters
batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

torch.manual_seed(SEED)

# =========================
# Data loading (kept in your style)
# =========================
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """Return a random batch of (X tokens, Y next-token targets) for train or val."""
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i + block_size] for i in ix])
    y = torch.stack([data_[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """Compute mean cross-entropy on train/val splits without gradient tracking."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# =========================
# Model (kept in your style)
# =========================
class Head(nn.Module):
    """Single head of self-attention with a causal mask."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)  # scaled dot-product attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # causal mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel, followed by a projection."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple 2-layer MLP with ReLU and dropout."""
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block: attention + feed-forward with pre-LN and residuals."""
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


class TransformerLM(nn.Module):
    """Tiny GPT-like language model."""
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)          # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                           # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits.view(B, T, -1), loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Sample tokens autoregressively, appending one step at a time."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =========================
# Pruning utilities (classic L1 magnitude pruning)
# =========================
def prunable_parameters(model):
    """
    Return a list of (module, 'weight') tuples that are eligible for pruning.
    Here we only prune nn.Linear weights (a common and simple starting point).
    Biases are not pruned.
    """
    params = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            params.append((m, 'weight'))
    return params

def apply_pruning(model, amount=0.5, global_prune=True):
    """
    Perform unstructured magnitude pruning using L1 norm:

    - If global_prune is True:
        Use prune.global_unstructured across all selected (module, 'weight') pairs.
        This shares a single threshold across layers. Weights with smallest absolute
        values (globally) are zeroed until the target fraction (amount) is reached.

    - If global_prune is False:
        Use prune.l1_unstructured per layer. Each layer prunes its own smallest
        absolute weights by the given fraction independently.

    Notes:
      • This function adds pruning reparameterizations to the modules:
           weight = weight_orig * weight_mask
        where weight_mask is a buffer (0/1). Gradients still flow to weight_orig.
      • After pruning, you typically fine-tune for some steps to recover accuracy.
    """
    params = prunable_parameters(model)
    if global_prune:
        prune.global_unstructured(
            params,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
    else:
        for (m, pname) in params:
            prune.l1_unstructured(m, name=pname, amount=amount)

def remove_pruning_reparam(model):
    """
    Permanently fold masks into the parameter tensors and remove the
    pruning reparameterization (weight_orig, weight_mask). After removal:
      • The module has a plain .weight parameter with zeros baked in.
      • This is recommended before exporting or saving the model for deployment.
    Safe to call even if some layers were not pruned.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            try:
                prune.remove(m, 'weight')
            except Exception:
                # Layer may not have pruning reparam; ignore and continue.
                pass

@torch.no_grad()
def report_sparsity(model, topk=10):
    """
    Print overall and per-layer sparsity for Linear weights:
      • Overall sparsity = (#zeros across all Linear weights) / (total elements)
      • Per-layer sparsity printed for the top-K most sparse layers.
    """
    layer_stats = []
    total_zeros, total_elems = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            W = m.weight
            zeros = (W == 0).sum().item()
            elems = W.numel()
            total_zeros += zeros
            total_elems += elems
            layer_stats.append((name, zeros/elems, W.shape))
    layer_stats.sort(key=lambda x: x[1], reverse=True)
    overall = total_zeros / max(1, total_elems)
    print(f"[Sparsity] Overall: {overall:.2%}  (zeros {total_zeros} / {total_elems})")
    print(f"[Sparsity] Top-{min(topk,len(layer_stats))} layers by sparsity:")
    for i, (n, sp, shp) in enumerate(layer_stats[:topk]):
        print(f"  #{i+1:<2} {n:<40} {str(tuple(shp)):<15}  {sp:.2%}")

# =========================
# Train → Prune → Finetune → Generate
# =========================
model = TransformerLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

did_prune = False
for iter in range(max_iters):

    # Periodic evaluation to track train/val loss
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        tag = " (after prune)" if did_prune else ""
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}{tag}")

    # Apply pruning exactly once at PRUNE_AT_ITER, then start fine-tuning
    if (not did_prune) and (iter == PRUNE_AT_ITER):
        print(f"\n>>> Applying {'GLOBAL' if GLOBAL_PRUNE else 'PER-LAYER'} L1 magnitude pruning "
              f"to target sparsity={TARGET_SPARSITY:.0%} at step {iter} ...")
        apply_pruning(model, amount=TARGET_SPARSITY, global_prune=GLOBAL_PRUNE)
        report_sparsity(model, topk=REPORT_TOPK)
        did_prune = True
        # Important: do NOT remove the pruning reparam now.
        # Keeping the mask active enforces zeros during fine-tuning.

    # One training (or fine-tuning) step
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # If pruning has been performed, stop after FINETUNE_ITERS additional steps
    if did_prune and (iter >= PRUNE_AT_ITER + FINETUNE_ITERS):
        print(f"\n>>> Finetune finished ({FINETUNE_ITERS} iters). "
              f"Folding masks (remove reparam) and stopping training.")
        remove_pruning_reparam(model)
        report_sparsity(model, topk=REPORT_TOPK)
        break

# If we never broke out (e.g., no pruning or early exit), still fold masks before export/use.
remove_pruning_reparam(model)
print("\n>>> Final sparsity report:")
report_sparsity(model, topk=REPORT_TOPK)

# Quick sample generation demo
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n=== Sample ===")
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
