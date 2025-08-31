import torch
import torch.nn as nn
from torch.nn import functional as F

# KIarpathy's hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from char to int
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
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


class Head(nn.Module):
    """single head of self-attention"""

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

        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)  # 建议写法
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

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
    """a simple linear layer followed b ya non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Attn论文中认为隐藏层应该是embedding size的四倍，所以*4
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # 变换回来
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

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


# super simple bigram model
class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = Head(n_embd)                               # 相比bigram新增 - 调用单头注意力机制
        # self.sa_head = MultiHeadAttention(4, n_embd//4)   # 相比上面那句换成了多头注意力
        # self.ffwd = FeedForward(n_embd)
        
        # Vanilla way to define block
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )
        
        #  a fancier way to do so
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        # x = self.sa_head(x) # 相比bigram新增 - 调用单头注意力机制
        # x = self.ffwd(x)
        x = self.blocks(x)    # 2025更新：这里已经用block代替了sahead和ffwd。
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_sizae)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # 因为pytorch需要另一个形状的输入，.view用于转置
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # generate函数的主要目的是利用训练好的模型进行文本生成
    def generate(self, idx, max_new_tokens):
        # idx is the (B, T) array of indices in the current context
        # 输入索引（idx）初始化：函数接收一个索引（idx），这通常表示当前文本的上下文或开始标记。这个索引通常是一个二维数组，其中第一维表示批次大小（B），第二维表示时间步（T）。

        # 循环生成：generate函数包含一个循环，循环次数由max_new_tokens参数决定，表示生成的最大新词元数量。在每次循环中，模型会根据当前的索引预测下一个词元的概率分布。
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # 相比bigram新增 - 这部分代码执行的操作是从每个批次中的序列 idx 选择最后 block_size 个词元。在 Python 和 PyTorch 中，索引 [:, -block_size:] 表示选择所有批次（:）中的每个序列的最后 block_size 个元素（-block_size:）。

            # get the predictions
            # 获取预测：在每个时间步，模型使用当前的索引调用前向传播（forward）函数来获取预测的对数概率（logits）和损失（loss）。然后只关注最后一个时间步的对数概率，因为我们只需要这一步的输出来决定下一个词元。
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)

            # get probabilities
            # 概率分布转换：使用softmax函数将对数概率转换为概率分布，这样就可以从中抽样下一个词元。
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            # 随机抽样，probs越大，抽到的概率越大
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            # 连续生成：更新后的索引被用于下一个循环的预测，这个过程重复进行直到生成了指定数量的新词元。
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = TransformerLM().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # typically 3e-4 but for small network lr=1e-3

for iter in range(max_iters):

    if iter % eval_interval == 0:
        
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)  # in other word, idx.
# 使用语言模型BigramLanguageModel来生成一个由300个新词元组成的文本序列，并将生成的索引转换成可读的文本。
# idx=torch.zeros((1, 1), dtype=torch.long)：这指定了生成文本的起始词元索引。这里使用了一个全为零的张量作为起始索引。通常，这个索引对应于某个特定的起始符号或词元，你需要根据你的词汇表设定它的实际含义。
# max_new_tokens=100：这指定了除了初始词元之外，你希望模型生成的最大新词元数量。
# model.generate(...)[0]：generate方法可能返回一个包含多个输出的元组（例如，索引和其他可能的状态信息）。这里的[0]是取出第一个元素，通常这应该是包含生成词元索引的张量。
# .tolist()：这一步将生成的词元索引张量转换成Python列表，以便进一步处理。
# decode(...)：上面存在一个名为decode的函数，它的功能是将词元索引列表转换成对应的字符串或可读的文本。
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
