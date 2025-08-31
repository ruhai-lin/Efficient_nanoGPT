import torch
import torch.nn as nn
from torch.nn import functional as F

# =======================
# Hyperparameters (kept close to the Base version)
# =======================
batch_size = 32
block_size = 16
max_iters = 2000                 # Max training steps for both Teacher and Student
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# ---- Student (same size as the Base model) ----
n_embd  = 64
n_head  = 4
n_layer = 4
dropout = 0.0

# ---- Teacher (only make it larger in width/depth/heads) ----
t_n_embd  = 128   # wider embedding size
t_n_head  = 8     # more attention heads (128//8 = 16 dims per head)
t_n_layer = 6     # deeper stack

# ---- KD settings ----
KD_T = 2.0        # temperature for softening logits
KD_ALPHA = 0.2    # blend factor between hard-label CE and soft-label KL
# =======================

torch.manual_seed(1337)

# ============== Data ==============
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
    """
    Sample a batch of contiguous token sequences of length `block_size`.
    Returns input tokens X and next-token targets Y, both on the selected device.
    """
    data_src = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i:i + block_size] for i in ix])
    y = torch.stack([data_src[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    """
    Evaluate mean cross-entropy over a few batches for both train/val splits.
    Used for quick progress reporting; keeps the model in eval mode temporarily.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

# ============== Minimal model changes: pass sizes explicitly so Teacher/Student can differ ==============
class Head(nn.Module):
    """
    A single self-attention head. Uses the provided embedding size instead of a global.
    This allows Teacher and Student to have different dimensions cleanly.
    """
    def __init__(self, head_size, n_embd_in):
        super().__init__()
        self.key   = nn.Linear(n_embd_in, head_size, bias=False)
        self.query = nn.Linear(n_embd_in, head_size, bias=False)
        self.value = nn.Linear(n_embd_in, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)  # scaled dot-product
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # causal mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention that concatenates head outputs and applies a projection."""
    def __init__(self, num_heads, n_embd_in):
        super().__init__()
        head_size = n_embd_in // num_heads
        self.heads = nn.ModuleList([Head(head_size, n_embd_in) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd_in, n_embd_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Simple 2-layer MLP with ReLU and dropout in the middle."""
    def __init__(self, n_embd_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd_in, 4 * n_embd_in),
            nn.ReLU(),
            nn.Linear(4 * n_embd_in, n_embd_in),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block with pre-LN, residuals, MHA, and MLP."""
    def __init__(self, n_embd_in, n_head_in):
        super().__init__()
        self.sa = MultiHeadAttention(n_head_in, n_embd_in)
        self.ffwd = FeedForward(n_embd_in)
        self.ln1 = nn.LayerNorm(n_embd_in)
        self.ln2 = nn.LayerNorm(n_embd_in)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    """
    Tiny GPT-style language model.
    Accepts embedding/head/layer sizes so we can instantiate a larger Teacher
    and a smaller Student with the same code path.
    """
    def __init__(self, n_embd_in=n_embd, n_head_in=n_head, n_layer_in=n_layer):
        super().__init__()
        self.n_embd_in = n_embd_in
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd_in)
        self.position_embedding_table = nn.Embedding(block_size, n_embd_in)
        self.blocks = nn.Sequential(*[Block(n_embd_in, n_head_in) for _ in range(n_layer_in)])
        self.ln_f = nn.LayerNorm(n_embd_in)
        self.lm_head = nn.Linear(n_embd_in, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                  # (B, T, C=n_embd_in)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                   # (B, T, V)
        loss = None
        if targets is not None:
            B_, T_, V_ = logits.shape
            loss = F.cross_entropy(logits.view(B_ * T_, V_), targets.view(B_ * T_))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Autoregressive sampling:
          1) Condition on the last `block_size` tokens.
          2) Predict next-token distribution.
          3) Sample one token and append to the sequence.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
# ============================================================

# ============== KD core (unchanged behavior, detailed explanation) ==============
def kd_loss(student_logits, teacher_logits, targets, T=2.0, alpha=0.2):
    """
    Knowledge Distillation loss that blends:
      • Hard-label CE (student vs. ground-truth) with weight alpha
      • Soft-label KL (student vs. teacher) with weight (1 - alpha)

    Steps:
      1) Soften both student and teacher logits by dividing by temperature T.
      2) Compute KL divergence between student log-probs and teacher probs.
      3) Multiply the KL by T^2 (standard scaling in KD).
      4) Compute standard cross-entropy between student's raw logits and labels.
      5) Return alpha * CE + (1 - alpha) * (T^2 * KL).

    Notes:
      - Higher T produces a softer probability distribution, exposing "dark knowledge"
        from the teacher about relative class similarities.
      - alpha near 0 leans on teacher guidance; alpha near 1 trusts hard labels more.
    """
    V = student_logits.size(-1)
    s = student_logits.view(-1, V)
    t = teacher_logits.view(-1, V)
    log_p_s = F.log_softmax(s / T, dim=-1)
    p_t     = F.softmax(t / T, dim=-1)
    kld = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)
    ce  = F.cross_entropy(s, targets.view(-1))
    return alpha * ce + (1 - alpha) * kld

# ============== Training ==============
# 1) Train a larger Teacher model with standard cross-entropy
teacher = TransformerLM(n_embd_in=t_n_embd, n_head_in=t_n_head, n_layer_in=t_n_layer).to(device)
optimizer_t = torch.optim.AdamW(teacher.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(teacher)
        print(f"[Teacher] step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = teacher(xb, yb)  # standard CE against hard labels
    optimizer_t.zero_grad(set_to_none=True)
    loss.backward()
    optimizer_t.step()

# 2) Train the smaller Student under KD:
#    - Teacher is put in eval mode (frozen).
#    - For each batch, we compute teacher logits (soft targets) and student logits.
#    - The KD loss blends hard-label CE and teacher-student KL with temperature.
student = TransformerLM().to(device)  # uses default student sizes (n_embd/n_head/n_layer)
optimizer_s = torch.optim.AdamW(student.parameters(), lr=learning_rate)
teacher.eval()  # freeze teacher for distillation

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(student)
        print(f"[Student] step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch('train')

    with torch.no_grad():
        t_logits, _ = teacher(xb)  # teacher soft targets (no grad)

    s_logits, _ = student(xb)
    loss = kd_loss(s_logits, t_logits, yb, T=KD_T, alpha=KD_ALPHA)

    optimizer_s.zero_grad(set_to_none=True)
    loss.backward()
    optimizer_s.step()

# ============== Sampling comparison (same style as Base) ==============
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n--- Teacher sample ---")
print(decode(teacher.generate(context.clone(), max_new_tokens=300)[0].tolist()))
print("\n--- Student sample ---")
print(decode(student.generate(context.clone(), max_new_tokens=300)[0].tolist()))
