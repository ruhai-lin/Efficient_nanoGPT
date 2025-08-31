import torch
import torch.nn as nn
from torch.nn import functional as F

# =========================
# Original hyperparameters (kept unchanged)
# =========================
batch_size = 8
block_size = 16
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

torch.manual_seed(1337)

# =========================
# Data loading (kept unchanged)
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
    """Return a random batch (X tokens, Y next-token targets) from train/val."""
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i + block_size] for i in ix])
    y = torch.stack([data_[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    """Evaluate average cross-entropy on train and val splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# =========================
# Model definition (kept same structure)
# =========================
class Head(nn.Module):
    """Single self-attention head with causal mask."""
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
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # causal mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention in parallel, followed by a projection."""
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
    """Two-layer MLP with ReLU and dropout."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: attention + MLP with pre-LN and residuals."""
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
    """A tiny GPT-style language model."""
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)              # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                               # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Greedy-free sampling: repeatedly sample next token and append."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =========================
# Training loop (kept unchanged)
# =========================
model = TransformerLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# =========================
# Minimal PTQ (SmoothQuant-style) → INT8 inference
# =========================

# Tunable knobs:
SMOOTH_ALPHA   = 0.6   # SmoothQuant alpha in [0, 1]; larger → shift more outliers from activations into weights
CALIB_SAMPLES  = 512   # How many tokens to run for calibration hooks (hundreds are enough for demo)
KEEP_FP        = []    # Optional list of substrings; if a module name contains any, skip quantization (keep in FP). e.g., ["lm_head"]

def iter_linears(root):
    """Iterate over all nn.Linear modules of the model."""
    for name, m in root.named_modules():
        if isinstance(m, nn.Linear):
            yield name, m

@torch.no_grad()
def collect_activation_max(model, num_tokens=CALIB_SAMPLES):
    """
    Calibration pass.
    For each Linear layer, record the per-input-channel absolute max of its incoming activations.
    Implementation detail:
      - Register a pre-forward hook on each Linear to see its input tensor.
      - Flatten batch/time dims, compute per-channel amax, and keep the running max.
    """
    model.eval()
    acts = {}   # map: Linear module -> tensor[in_features] of per-channel abs-max
    hooks = []

    for name, lin in iter_linears(model):
        if any(k in name for k in KEEP_FP):
            continue
        in_features = lin.in_features
        acts[lin] = torch.zeros(in_features, device=device)

        def make_hook(mod):
            def pre_hook(mod, inp):
                x = inp[0].detach()               # shape (..., in_features)
                x = x.reshape(-1, x.shape[-1])   # flatten batch/time to [N, in_features]
                m = x.abs().amax(dim=0)          # per-channel amax over N
                acts[mod] = torch.max(acts[mod], m)  # running max
            return pre_hook
        hooks.append(lin.register_forward_pre_hook(make_hook(lin)))

    # Run a few batches just to trigger the hooks and collect statistics
    seen = 0
    while seen < num_tokens:
        X, _ = get_batch('train')
        model(X)  # triggers hooks
        seen += X.numel()

    for h in hooks:
        h.remove()
    return acts

@torch.no_grad()
def prepare_sq_and_qparams(model, act_max_map, alpha=SMOOTH_ALPHA, eps=1e-8):
    """
    SmoothQuant parameter baking + quantization parameter extraction.
    For each Linear layer:
      1) Compute SmoothQuant scaling 's' per input channel based on
         activation max (a_max) and weight column max (w_col_max).
         s[j] = a_max[j]^alpha / w_col_max[j]^(1-alpha)
      2) Multiply weight columns by s (push some activation outliers into weights).
      3) Activation side will be divided by s at inference (we store 1/s).
      4) Choose per-tensor activation scale s_x using amax(a_max / s).
      5) Choose per-output-channel weight scales s_w using amax(|W|) per row.
      6) Create int8 packed weight W_int8 = round(clamp(W / s_w)).
    Returns a metadata dict with all tensors needed to build QuantLinearINT8.
    """
    meta = {}  # map: Linear -> dict of quantization artifacts
    for name, lin in iter_linears(model):
        if any(k in name for k in KEEP_FP):
            continue

        W = lin.weight.data.clone()                          # [out, in]
        bias = lin.bias.data.clone() if lin.bias is not None else None
        a_max = act_max_map[lin].clone() + eps               # [in]
        w_col_max = W.abs().amax(dim=0) + eps                # [in]

        # SmoothQuant balance factor per input channel
        s = (a_max.pow(alpha) / w_col_max.pow(1.0 - alpha)).clamp(min=eps)   # [in]

        # Apply column-wise scaling to weights: W[:, j] *= s[j]
        W.mul_(s)

        # After balancing, the per-tensor activation scale can be derived from amax(a_max / s)
        x_bal_max = (a_max / s).amax()                       # scalar
        s_x = (x_bal_max / 127.0).clamp(min=eps).item()      # scalar (float), for per-tensor activation quant

        # Weight per-output-channel symmetric int8 quantization
        w_row_max = W.abs().amax(dim=1) + eps                # [out]
        s_w = (w_row_max / 127.0).clamp(min=eps)             # [out]
        W_int8 = torch.round((W / s_w.unsqueeze(1)).clamp(-127, 127)).to(torch.int8)  # [out, in]

        meta[lin] = {
            "act_balance_inv": (1.0 / s).detach(),           # [in] store 1/s to divide activations at inference
            "s_x": float(s_x),                               # scalar activation scale
            "s_w": s_w.detach(),                             # [out] weight scales per output channel
            "w_int8": W_int8.detach().contiguous(),          # packed int8 weights
            "bias": bias.detach() if bias is not None else None,
            "name": name,
        }
    return meta

class QuantLinearINT8(nn.Module):
    """
    Inference-only int8 linear layer with SmoothQuant balancing.

    Computation:
      Let x be the floating-point input of shape [..., in_features].
      1) Balance activations: x' = x * act_balance_inv  (this divides each channel by s).
      2) Quantize activations per-tensor: x_int = round(clamp(x'/s_x, -127, 127)).
      3) int8 GEMM: acc = x_int @ W_int8^T  (accumulates into fp32 here for simplicity).
      4) Dequantize: y = acc * (s_x * s_w) + bias.

    Notes:
      - We keep scales and int8 weights as buffers.
      - We reshape to 2D for GEMM and then restore the original shape in the output.
    """
    def __init__(self, qmeta, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("w_int8", qmeta["w_int8"])                  # [out, in] int8
        self.register_buffer("s_w",    qmeta["s_w"].to(torch.float32))   # [out] fp32 scales
        self.register_buffer("act_balance_inv", qmeta["act_balance_inv"].to(torch.float32))  # [in]
        self.s_x = float(qmeta["s_x"])                                   # python float
        if qmeta["bias"] is not None:
            self.register_buffer("bias", qmeta["bias"].to(torch.float32))
        else:
            self.bias = None

    def forward(self, x):
        # x has shape [..., in_features]; flatten to [N, in] for matmul
        orig_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])                       # [N, in]

        # SmoothQuant balancing: divide per-channel by s using pre-stored (1/s)
        x_bal = x * self.act_balance_inv.to(x.device)     # [N, in]

        # Per-tensor activation quantization to int8
        s_x = torch.tensor(self.s_x, dtype=torch.float32, device=x.device)
        x_int = torch.round((x_bal / s_x).clamp(-127, 127)).to(torch.int8)  # [N, in]

        # int8 @ int8 → accumulate as float32 (simple demo; kernels often use int32 acc)
        acc = torch.matmul(x_int.to(torch.float32), self.w_int8.to(torch.float32).t())  # [N, out]

        # Dequantize: multiply by s_x * s_w
        y = acc * (s_x * self.s_w.to(x.device))

        if self.bias is not None:
            y = y + self.bias.to(x.device)

        return y.view(*orig_shape, self.out_features)

def quantize_model_int8(model, keep_fp=KEEP_FP):
    """
    Top-level PTQ routine:
      - Collect activation stats via hooks.
      - Prepare SmoothQuant scaling and quant parameters.
      - Recursively replace nn.Linear with QuantLinearINT8 (unless excluded).
    """
    acts = collect_activation_max(model, CALIB_SAMPLES)
    meta = prepare_sq_and_qparams(model, acts, SMOOTH_ALPHA)

    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                # Optional: skip layers whose names match KEEP_FP entries
                if any(k in name for k in keep_fp):
                    continue
                if child in meta:
                    q = QuantLinearINT8(meta[child], child.in_features, child.out_features)
                    setattr(module, name, q)
            else:
                replace(child)
    replace(model)
    return model

def export_int8_artifacts(model, path="export_int8.pth"):
    """
    Export a minimal package for deployment:
      - int8 weights
      - per-output-channel weight scales (s_w)
      - per-tensor activation scale (s_x)
      - per-channel activation balancing factors (act_balance_inv)
      - optional bias
    Keys are module names to facilitate reconstruction in an external runtime.
    """
    pkg = {}
    for name, m in model.named_modules():
        if isinstance(m, QuantLinearINT8):
            pkg[name] = {
                "weight_int8": m.w_int8.cpu(),
                "s_w": m.s_w.cpu(),
                "s_x": m.s_x,
                "act_balance_inv": m.act_balance_inv.cpu(),
                "bias": None if m.bias is None else m.bias.cpu(),
                "in_features": m.in_features,
                "out_features": m.out_features,
            }
    torch.save(pkg, path)
    print(f"[INT8 Export] saved to {path} with {len(pkg)} layers.")

# ======== Quantize + reporting (calibration, memory, speed, similarity, loss) ========
import copy, time

def human_bytes(n):
    """Format bytes into a human-readable string (KB/MB/GB...)."""
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024.0: return f"{n:.2f} {u}"
        n /= 1024.0
    return f"{n:.2f} PB"

def fp32_param_bytes(model):
    """Total parameter bytes of an FP32 model (weights + biases that are Parameters)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())

def int8_weight_bytes(model_int8):
    """
    Approximate memory footprint for the quantized model:
      - Parameters that remain as fp32 (non-replaced modules).
      - Buffers used by QuantLinearINT8: int8 weights, fp32 scales, fp32 bias, fp32 balance.
    This gives an apples-to-apples rough weight storage comparison against FP32.
    """
    other_param_bytes = sum(p.numel() * p.element_size() for p in model_int8.parameters())
    qbuf_bytes = 0
    for m in model_int8.modules():
        if isinstance(m, QuantLinearINT8):
            qbuf_bytes += m.w_int8.numel() * 1      # int8 = 1 byte
            qbuf_bytes += m.s_w.numel() * 4         # fp32 scales
            qbuf_bytes += m.act_balance_inv.numel() * 4
            if m.bias is not None:
                qbuf_bytes += m.bias.numel() * 4
    return other_param_bytes + qbuf_bytes

@torch.no_grad()
def measure_peak_mem_forward(model, iters=50):
    """
    Run several forward passes and record CUDA peak allocated memory.
    Returns None on CPU-only, otherwise an integer number of bytes.
    """
    if not torch.cuda.is_available():
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    for _ in range(iters):
        X, Y = get_batch('val')
        model(X, Y)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device)

@torch.no_grad()
def measure_gen_speed(model, tokens=200):
    """
    Measure simple generation throughput (tokens/s) for a fixed-length sample.
    Synchronizes CUDA to get a stable wall-clock measurement.
    """
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model.generate(context, max_new_tokens=tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return tokens / max(dt, 1e-8)

@torch.no_grad()
def capture_layer_outputs(model, names, layer_type):
    """
    Capture and flatten the outputs of specified layers.
    This helps compare FP32 vs INT8 numerically (cosine/MSE).
    We cap the length to avoid huge tensors in printing.
    """
    outs = {}
    hooks = []
    moddict = dict(model.named_modules())

    def make_hook(n):
        def hook(mod, inp, out):
            v = out.detach().float().reshape(-1)
            outs[n] = v[:65536].clone()  # keep only first 65k elements
        return hook

    for n in names:
        m = moddict.get(n, None)
        if m is not None and isinstance(m, layer_type):
            hooks.append(m.register_forward_hook(make_hook(n)))

    X, _ = get_batch('val')
    model(X)  # trigger hooks

    for h in hooks: h.remove()
    return outs

def compare_layers(fp32_model, int8_model, topk=5):
    """
    Compare layer outputs between FP32 and INT8 models:
      - Pick all original Linear layer names.
      - Grab their outputs (FP32 vs QuantLinearINT8).
      - Compute cosine similarity and MSE for each.
      - Return the worst K layers by cosine similarity.
    """
    lin_names = [n for n,m in fp32_model.named_modules() if isinstance(m, nn.Linear)]
    outs_fp32 = capture_layer_outputs(fp32_model, lin_names, nn.Linear)
    outs_int8 = capture_layer_outputs(int8_model, lin_names, QuantLinearINT8)

    rows = []
    for n in lin_names:
        a = outs_fp32.get(n, None)
        b = outs_int8.get(n, None)
        if a is None or b is None or a.numel()==0 or b.numel()==0:
            continue
        L = min(a.numel(), b.numel())
        a = a[:L]; b = b[:L]
        cos = F.cosine_similarity(a, b, dim=0).item()
        mse = F.mse_loss(a, b).item()
        rows.append((n, cos, mse))
    rows.sort(key=lambda x: x[1])  # ascending by cosine
    return rows[:topk], rows

print("\n>>> Preparing FP32 baseline & converting to INT8 (SmoothQuant-style PTQ)...")
model_fp32 = copy.deepcopy(model).to(device).eval()

# Quantize a deep-copied model so the FP32 baseline remains intact
model_int8 = quantize_model_int8(copy.deepcopy(model).to(device))

# 1) Parameter/weight memory comparison
fp32_bytes = fp32_param_bytes(model_fp32)
int8_bytes = int8_weight_bytes(model_int8)
print("\n[Model Weights]")
print(f"  FP32 total params: {human_bytes(fp32_bytes)}")
print(f"  INT8 packed (weights+scales+bias+balance): {human_bytes(int8_bytes)}")
if int8_bytes > 0:
    print(f"  Compression ratio: {fp32_bytes / int8_bytes:.2f}x")

# 2) Peak inference memory (forward passes)
pk_fp32 = measure_peak_mem_forward(model_fp32, iters=50)
pk_int8 = measure_peak_mem_forward(model_int8,  iters=50)
if pk_fp32 is not None and pk_int8 is not None:
    print("\n[Peak CUDA Memory (forward x50)]")
    print(f"  FP32: {human_bytes(pk_fp32)}")
    print(f"  INT8: {human_bytes(pk_int8)}")
    print(f"  Reduction: { (1 - pk_int8/max(pk_fp32,1)) * 100:.2f}%")

# 3) Generation throughput (tokens/s)
tp_fp32 = measure_gen_speed(model_fp32, tokens=200)
tp_int8 = measure_gen_speed(model_int8,  tokens=200)
print("\n[Throughput: generate 200 tokens]")
print(f"  FP32: {tp_fp32:.2f} tokens/s")
print(f"  INT8: {tp_int8:.2f} tokens/s")

# 4) Numerical consistency: layer-wise cosine/MSE
worst_k, _ = compare_layers(model_fp32, model_int8, topk=5)
print("\n[Layer-wise similarity (worst 5 by cosine)]")
for n, cos, mse in worst_k:
    print(f"  {n:<45} cos={cos:.5f}  mse={mse:.3e}")

# 5) Validation loss comparison (re-uses estimate_loss)
losses_fp32 = estimate_loss(model_fp32)
losses_int8 = estimate_loss(model_int8)
print("\n[Val Loss]")
print(f"  FP32: {losses_fp32['val']:.4f}")
print(f"  INT8: {losses_int8['val']:.4f}")

# Export artifacts + sample generation with INT8 model
export_int8_artifacts(model_int8, "export_int8.pth")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n=== INT8 Sample ===")
print(decode(model_int8.generate(context, max_new_tokens=300)[0].tolist()))
