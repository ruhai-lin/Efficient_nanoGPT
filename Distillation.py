# ============================================
# Distillation.py — Fully Commented (Didactic)
# Matches the Base style; only adds KD-specific logic & size decoupling.
# ============================================

import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------
# Hyperparameters
# -------------------------------
batch_size   = 16   # larger batch than Base to stabilize KD statistics if desired
block_size   = 32   # context length (T). Must be the SAME for teacher & student here.
max_iters    = 1000 # was 5000 in Base.py, reduced for demo purpose
eval_interval= 100
learning_rate= 1e-3
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters   = 200

# -------------------------------
# Student config (Same as Base)
# -------------------------------
n_embd  = 64
n_head  = 4
n_layer = 4
dropout = 0.0  # keep disabled for tiny models for faster convergence

# -------------------------------
# Teacher config (bigger than Student)
# Only width/heads/depth are increased; same tokenizer/vocab/block_size.
# -------------------------------
t_n_embd  = 128  # 2x wider model dimension
t_n_head  = 8    # 2x more heads; also ensures head_size = 128//8 = 16
t_n_layer = 6    # deeper stack

# -------------------------------
# KD (Knowledge Distillation) settings
# -------------------------------
KD_T     = 2.0  # Temperature: >1 softens teacher distribution; typical range 1.5~4.0 for classification-like tasks
KD_ALPHA = 0.2  # Mixing weight for hard-label CE vs. soft-label KL:
                #   final_loss = alpha * CE(student, hard) + (1 - alpha) * (T^2 * KL(student_T || teacher_T))
                # Intuition:
                #   - alpha close to 1: trust ground-truth labels more (less teacher guidance)
                #   - alpha close to 0: rely more on teacher's "dark knowledge"
                # Valid range: [0, 1]. In practice 0.1~0.7 are common; tune jointly with KD_T.
                # Note: Multiplying KL by T^2 preserves gradient magnitudes per standard KD (Hinton et al.).
torch.manual_seed(1337)

# =======================
# Very similar to Base.py
# =======================
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]                     # str -> List[int]
decode = lambda l: ''.join([itos[i] for i in l])            # List[int] -> str

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

def get_batch(split: str):
    data_src = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i : i + block_size]         for i in ix])
    y = torch.stack([data_src[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model: nn.Module): # Same as Base but parameterized by input `model`
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ============================================
# Transformer (dimension-parameterized)
# Only difference vs. Base: we pass sizes explicitly (so teacher/student can differ).
# ============================================

class Head(nn.Module):
    """
    Single self-attention head.
    Differences vs. Base:
      - Use `n_embd_in` (passed in) instead of a global `n_embd`.
      - Keep the same causal masking strategy with a prebuilt lower-triangular matrix.
    Shapes:
      x: (B, T, C=n_embd_in) -> out: (B, T, head_size)
    """
    def __init__(self, head_size: int, n_embd_in: int):
        super().__init__()
        self.key   = nn.Linear(n_embd_in, head_size, bias=False)
        self.query = nn.Linear(n_embd_in, head_size, bias=False)
        self.value = nn.Linear(n_embd_in, head_size, bias=False)
        # Note: mask is created for the global `block_size`. If you later change block_size,
        #       ensure teacher & student use the same value, or rebuild masks accordingly.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)                # (B, T, head_size)
        q = self.query(x)              # (B, T, head_size)
        # Scaled dot-product attention scores
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)  # (B, T, T)
        # Causal mask: disallow attending to future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)   # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)              # (B, T, head_size)
        out = wei @ v                  # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, n_embd_in: int):
        super().__init__()
        # Assumes `n_embd_in` divisible by `num_heads`
        head_size = n_embd_in // num_heads
        self.heads = nn.ModuleList([Head(head_size, n_embd_in) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd_in, n_embd_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C)
        out = self.dropout(self.proj(out))                   # (B, T, C)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd_in, 4 * n_embd_in),
            nn.ReLU(),
            nn.Linear(4 * n_embd_in, n_embd_in),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd_in: int, n_head_in: int):
        super().__init__()
        self.sa   = MultiHeadAttention(n_head_in, n_embd_in)
        self.ffwd = FeedForward(n_embd_in)
        self.ln1  = nn.LayerNorm(n_embd_in)
        self.ln2  = nn.LayerNorm(n_embd_in)

    def forward(self, x: torch.Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, n_embd_in=n_embd, n_head_in=n_head, n_layer_in=n_layer):
        super().__init__()
        self.n_embd_in = n_embd_in
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd_in)  # (V -> C)
        self.position_embedding_table = nn.Embedding(block_size, n_embd_in)  # (T -> C)
        self.blocks = nn.Sequential(*[Block(n_embd_in, n_head_in) for _ in range(n_layer_in)])
        self.ln_f   = nn.LayerNorm(n_embd_in)
        self.lm_head= nn.Linear(n_embd_in, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                          # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb                                              # (B, T, C)
        x = self.blocks(x)                                                 # (B, T, C)
        x = self.ln_f(x)                                                   # (B, T, C)
        logits = self.lm_head(x)                                           # (B, T, V)

        loss = None
        if targets is not None:
            B_, T_, V_ = logits.shape
            # Flatten (B,T) to (B*T,) for CE
            loss = F.cross_entropy(logits.view(B_ * T_, V_), targets.view(B_ * T_))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                 # (B, V)
            probs  = F.softmax(logits, dim=-1)        # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)   # (B, T+1)
        return idx

# ============================================
# KD objective (core of this file)
# ============================================
def kd_loss(student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            targets: torch.Tensor,
            T: float = 2.0,
            alpha: float = 0.2) -> torch.Tensor:
    """
    Knowledge Distillation loss that blends:
      (1) Hard-label CE:      CE(student_logits, targets)
      (2) Soft-label KD term: KL( softmax(student_logits/T) || softmax(teacher_logits/T) )

    Returned loss:
      alpha * CE + (1 - alpha) * (T^2 * KL)

    Why T^2?
      - With temperature scaling, gradients scale as 1/T^2 unless compensated.
        Multiplying by T^2 restores comparable magnitudes (per standard KD practice).

    Practical tips:
      - T > 1 softens distributions, revealing teacher's “dark knowledge” about near-misses.
      - If teacher is much stronger, you can lower alpha (e.g., 0.1~0.3) to lean on teacher more.
      - If labels are noisy, prefer smaller alpha. If teacher is weak, increase alpha.

    Shapes:
      student_logits, teacher_logits: (B, T, V)
      targets:                        (B, T)

    Notes:
      - We compute KL over the flattened (B*T, V).
      - KL takes log-probs of the student and probs of the teacher.
    """
    V = student_logits.size(-1)
    s = student_logits.view(-1, V)  # (B*T, V)
    t = teacher_logits.view(-1, V)  # (B*T, V)

    # Temperature-softened distributions
    log_p_s = F.log_softmax(s / T, dim=-1)  # student log-probs at temperature T
    p_t     = F.softmax(t / T, dim=-1)      # teacher probs at temperature T

    # KL(student_T || teacher_T), averaged over the batch
    kld = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)

    # Standard CE with hard labels (no temperature)
    ce  = F.cross_entropy(s, targets.view(-1))

    # Final blended loss
    return alpha * ce + (1.0 - alpha) * kld

# ============================================
# Phase 1: Train the TEACHER with standard CE
# ============================================
teacher = TransformerLM(
    n_embd_in=t_n_embd,
    n_head_in=t_n_head,
    n_layer_in=t_n_layer
).to(device)

optimizer_t = torch.optim.AdamW(teacher.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(teacher)
        print(f"[Teacher] step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = teacher(xb, yb)  # standard CE loss against hard labels
    optimizer_t.zero_grad(set_to_none=True)
    loss.backward()
    optimizer_t.step()

# ============================================
# Phase 2: Train the STUDENT under KD
#   - Freeze teacher (eval mode, no grad)
#   - For each batch: compute teacher logits (soft targets) and student logits
#   - Optimize KD loss
# ============================================
student = TransformerLM(  # defaults to the student sizes defined above
    n_embd_in=n_embd,
    n_head_in=n_head,
    n_layer_in=n_layer
).to(device)

optimizer_s = torch.optim.AdamW(student.parameters(), lr=learning_rate)
teacher.eval()  # IMPORTANT: disable dropout etc.; also implies we won't update teacher

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(student)
        print(f"[Student] step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch('train')

    with torch.no_grad():
        t_logits, _ = teacher(xb)   # (B, T, V); teacher remains frozen

    s_logits, _ = student(xb)       # (B, T, V)
    loss = kd_loss(s_logits, t_logits, yb, T=KD_T, alpha=KD_ALPHA)

    optimizer_s.zero_grad(set_to_none=True)
    loss.backward()
    optimizer_s.step()

# ------------------------------
# Text generation demo
# ------------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)

print("\n--- Teacher sample ---")
print(decode(teacher.generate(context.clone(), max_new_tokens=300)[0].tolist()))

print("\n--- Student sample ---")
print(decode(student.generate(context.clone(), max_new_tokens=300)[0].tolist()))
