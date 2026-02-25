# Transformer Deep Dive: Step-by-Step

This chapter walks through the Transformer architecture in detail, explaining each component with simple examples and concrete numbers. By the end, you'll understand exactly what happens when text flows through a Transformer.

---

## Overview: The Big Picture

The Transformer processes text in these steps:

```
Input Text
    ↓
1. Tokenization      → Convert text to token IDs
    ↓
2. Embedding         → Convert IDs to vectors
    ↓
3. Positional Encoding → Add position information
    ↓
4. Transformer Blocks (× N layers)
   ├── Self-Attention  → Tokens communicate with each other
   ├── Add & Norm      → Residual connection + normalization
   ├── Feed-Forward    → Process each token independently
   └── Add & Norm      → Residual connection + normalization
    ↓
5. Output Layer      → Convert to vocabulary probabilities
    ↓
Output Token
```

**Running example**: We'll trace `"The cat sat"` through a tiny Transformer with:
- Vocabulary size: 10,000
- Embedding dimension (d_model): 512
- Number of attention heads: 8
- Feed-forward hidden dimension: 2048
- Number of layers: 6

---

## Step 1: Tokenization

### What happens

Text is split into tokens and converted to integer IDs.

```
Input:  "The cat sat"
Tokens: ["The", " cat", " sat"]
IDs:    [464, 3797, 3332]
```

### Why it matters

- Neural networks need numbers, not strings
- Subword tokenization (BPE) handles rare words by splitting them
- The vocabulary is fixed at training time

### Concrete example

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

text = "The cat sat"
tokens = enc.encode(text)
print(tokens)  # [791, 8415, 7731]
print([enc.decode([t]) for t in tokens])  # ['The', ' cat', ' sat']
```

**Output shape**: `(batch_size, seq_len)` = `(1, 3)` — three token IDs

---

## Step 2: Token Embedding

### What happens

Each token ID is converted to a dense vector by looking up a learned embedding table.

```
Token IDs:     [464,    3797,   3332]
                ↓        ↓       ↓
Embeddings:   [e₁,      e₂,     e₃]
              (512,)   (512,)  (512,)
```

### The math

```python
# Embedding table: (vocab_size, d_model) = (10000, 512)
embedding_table = nn.Embedding(10000, 512)

# Lookup
token_ids = torch.tensor([464, 3797, 3332])  # (3,)
embeddings = embedding_table(token_ids)       # (3, 512)
```

### Why it matters

- Embeddings capture semantic meaning
- Similar words have similar embeddings
- These embeddings are learned during training

**Output shape**: `(batch_size, seq_len, d_model)` = `(1, 3, 512)`

---

## Step 3: Positional Encoding

### The problem

Self-attention is **permutation-invariant** — it doesn't know word order!

```
"cat sat mat" and "mat sat cat" would produce identical attention weights
```

### The solution

Add position information to each embedding.

```
Final embedding = Token embedding + Positional encoding
```

### Sinusoidal positional encoding (original Transformer)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` = position in sequence (0, 1, 2, ...)
- `i` = dimension index (0, 1, 2, ..., d_model/2)

### Concrete example

```python
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]      # (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]        # (1, d_model)
    
    # Compute angles
    angles = pos / (10000 ** (2 * (i // 2) / d_model))
    
    # Apply sin to even indices, cos to odd indices
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])  # Even dimensions
    pe[:, 1::2] = np.cos(angles[:, 1::2])  # Odd dimensions
    
    return pe

pe = positional_encoding(3, 512)  # (3, 512)
```

### Why sinusoidal?

1. **Unique per position**: Each position gets a distinct encoding
2. **Relative positions**: `PE(pos+k)` can be expressed as a linear function of `PE(pos)`
3. **Generalizes**: Works for sequences longer than training data

### Learned positional embeddings (GPT, BERT)

```python
# Alternative: Learn position embeddings
position_embedding = nn.Embedding(max_seq_len, d_model)
positions = torch.arange(seq_len)  # [0, 1, 2]
pos_emb = position_embedding(positions)  # (3, 512)
```

**Output shape**: `(1, 3, 512)` — same as input, but now position-aware

---

## Step 4: Self-Attention

This is the core innovation of the Transformer. Each token looks at all other tokens to gather relevant information.

### The intuition

For the sentence "The cat sat on the mat":
- When processing "sat", the model should pay attention to "cat" (the subject)
- When processing "mat", the model should pay attention to "on" and "the"

### Query, Key, Value

Each token produces three vectors:

| Vector | Meaning | Analogy |
|--------|---------|---------|
| **Query (Q)** | "What am I looking for?" | A search query |
| **Key (K)** | "What do I contain?" | A document title |
| **Value (V)** | "What information do I provide?" | Document content |

### The math

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

### Step-by-step with numbers

Let's trace through with d_model = 4 (simplified):

**Input**: 3 tokens, each with 4-dimensional embedding

```
X = [[0.1, 0.2, 0.3, 0.4],   # "The"
     [0.5, 0.6, 0.7, 0.8],   # "cat"
     [0.2, 0.3, 0.4, 0.5]]   # "sat"
```

**Step 4.1: Create Q, K, V**

```python
# Learned weight matrices (4x4 each)
W_Q = [[...], [...], [...], [...]]  # (d_model, d_k)
W_K = [[...], [...], [...], [...]]  # (d_model, d_k)
W_V = [[...], [...], [...], [...]]  # (d_model, d_v)

Q = X @ W_Q  # (3, 4) @ (4, 4) = (3, 4)
K = X @ W_K  # (3, 4)
V = X @ W_V  # (3, 4)
```

Result (example values):
```
Q = [[0.2, 0.1, 0.3, 0.2],   # Query for "The"
     [0.5, 0.4, 0.6, 0.3],   # Query for "cat"
     [0.3, 0.2, 0.4, 0.3]]   # Query for "sat"

K = [[0.1, 0.2, 0.2, 0.1],   # Key for "The"
     [0.4, 0.5, 0.3, 0.2],   # Key for "cat"
     [0.2, 0.3, 0.3, 0.2]]   # Key for "sat"

V = [[0.3, 0.1, 0.2, 0.4],   # Value for "The"
     [0.6, 0.2, 0.3, 0.5],   # Value for "cat"
     [0.4, 0.2, 0.3, 0.4]]   # Value for "sat"
```

**Step 4.2: Compute attention scores**

```python
scores = Q @ K.T  # (3, 4) @ (4, 3) = (3, 3)
```

```
scores = [[0.14, 0.36, 0.24],   # "The" attending to [The, cat, sat]
          [0.22, 0.58, 0.38],   # "cat" attending to [The, cat, sat]
          [0.18, 0.46, 0.30]]   # "sat" attending to [The, cat, sat]
```

**Step 4.3: Scale by √d_k**

```python
d_k = 4
scaled_scores = scores / np.sqrt(d_k)  # Divide by 2
```

```
scaled_scores = [[0.07, 0.18, 0.12],
                 [0.11, 0.29, 0.19],
                 [0.09, 0.23, 0.15]]
```

**Why scale?** Without scaling, large dot products cause softmax to saturate (output ~0 or ~1), leading to tiny gradients.

**Step 4.4: Apply softmax (row-wise)**

```python
attention_weights = softmax(scaled_scores, axis=-1)
```

```
attention_weights = [[0.31, 0.37, 0.32],   # "The" attention distribution
                     [0.30, 0.40, 0.30],   # "cat" attention distribution
                     [0.31, 0.38, 0.31]]   # "sat" attention distribution
```

Each row sums to 1. "cat" gets the most attention from all tokens (highest scores).

**Step 4.5: Weighted sum of values**

```python
output = attention_weights @ V  # (3, 3) @ (3, 4) = (3, 4)
```

```
output = [[0.43, 0.17, 0.27, 0.43],   # New representation for "The"
          [0.44, 0.17, 0.27, 0.44],   # New representation for "cat"
          [0.43, 0.17, 0.27, 0.43]]   # New representation for "sat"
```

Each token now contains information from all tokens, weighted by relevance!

### Causal masking (for decoder/GPT)

In autoregressive models, tokens can only attend to previous tokens:

```
Before softmax, add mask:
          The   cat   sat
The    [  0    -∞    -∞  ]
cat    [  0     0    -∞  ]
sat    [  0     0     0  ]
```

After softmax, -∞ becomes 0:
```
          The   cat   sat
The    [ 1.0   0.0   0.0 ]   # "The" only sees itself
cat    [ 0.45  0.55  0.0 ]   # "cat" sees "The" and itself
sat    [ 0.31  0.38  0.31]   # "sat" sees all previous
```

**Output shape**: `(1, 3, 512)` — same as input

---

## Step 5: Multi-Head Attention

### The problem with single attention

One attention head can only focus on one type of relationship at a time.

### The solution

Run multiple attention heads in parallel, each learning different patterns:

```
Head 1: Syntactic relationships (subject-verb)
Head 2: Semantic relationships (synonyms)
Head 3: Positional relationships (nearby words)
Head 4: Coreference (pronouns → nouns)
...
```

### The math

```python
# Split d_model into h heads
d_model = 512
n_heads = 8
d_k = d_model // n_heads  # 64 per head

# Each head has its own Q, K, V projections
for i in range(n_heads):
    Q_i = X @ W_Q[i]  # (seq_len, 64)
    K_i = X @ W_K[i]  # (seq_len, 64)
    V_i = X @ W_V[i]  # (seq_len, 64)
    head_i = Attention(Q_i, K_i, V_i)  # (seq_len, 64)

# Concatenate all heads
concat = [head_1, head_2, ..., head_8]  # (seq_len, 512)

# Final projection
output = concat @ W_O  # (seq_len, 512)
```

### Efficient implementation

In practice, we do this with a single matrix multiplication:

```python
# Project all heads at once
Q = X @ W_Q  # (batch, seq_len, 512)
K = X @ W_K
V = X @ W_V

# Reshape to (batch, n_heads, seq_len, d_k)
Q = Q.reshape(batch, seq_len, n_heads, d_k).transpose(1, 2)
K = K.reshape(batch, seq_len, n_heads, d_k).transpose(1, 2)
V = V.reshape(batch, seq_len, n_heads, d_k).transpose(1, 2)

# Attention for all heads in parallel
# (batch, n_heads, seq_len, d_k) @ (batch, n_heads, d_k, seq_len)
# = (batch, n_heads, seq_len, seq_len)
scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
weights = softmax(scores)
output = weights @ V  # (batch, n_heads, seq_len, d_k)

# Reshape back and project
output = output.transpose(1, 2).reshape(batch, seq_len, d_model)
output = output @ W_O
```

**Output shape**: `(1, 3, 512)` — same as input

---

## Step 6: Add & Norm (Residual Connection + Layer Normalization)

### Residual connection

```python
output = attention_output + input  # Skip connection
```

**Why?**
- Allows gradients to flow directly through the network
- Helps train very deep networks (6+ layers)
- The network can learn "do nothing" by outputting zeros

### Layer normalization

```python
# Normalize across the feature dimension (d_model)
mean = output.mean(dim=-1, keepdim=True)
std = output.std(dim=-1, keepdim=True)
normalized = (output - mean) / (std + eps)

# Learnable scale and shift
output = gamma * normalized + beta
```

**Why layer norm instead of batch norm?**
- Batch norm: normalizes across batch (depends on other samples)
- Layer norm: normalizes across features (independent per sample)
- For variable-length sequences, layer norm is more stable

### Pre-norm vs Post-norm

**Post-norm (original Transformer)**:
```
output = LayerNorm(x + Attention(x))
```

**Pre-norm (GPT-2, modern)**:
```
output = x + Attention(LayerNorm(x))
```

Pre-norm is easier to train and more stable for deep networks.

**Output shape**: `(1, 3, 512)` — same as input

---

## Step 7: Feed-Forward Network (FFN)

### What happens

Each token passes through a two-layer MLP independently:

```python
def feed_forward(x):
    # x: (seq_len, 512)
    hidden = relu(x @ W1 + b1)  # (seq_len, 2048) - expand
    output = hidden @ W2 + b2   # (seq_len, 512)  - contract
    return output
```

### The dimensions

```
Input:  (seq_len, 512)
    ↓
W1:     (512, 2048)     # Expand 4x
    ↓
Hidden: (seq_len, 2048)
    ↓
W2:     (2048, 512)     # Contract back
    ↓
Output: (seq_len, 512)
```

### Why expand then contract?

- The expanded dimension (2048) provides more capacity
- Think of it as: attention gathers information, FFN processes it
- Each token "thinks" about what it learned from attention

### Activation functions

| Activation | Formula | Used in |
|------------|---------|---------|
| ReLU | max(0, x) | Original Transformer |
| GELU | x × Φ(x) | GPT-2, BERT |
| SwiGLU | Swish(xW) × (xV) | LLaMA, PaLM |

**GELU** (Gaussian Error Linear Unit) is smoother than ReLU:
```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

### Another Add & Norm

After FFN, another residual connection and layer norm:

```python
output = LayerNorm(ffn_output + attention_output)
```

**Output shape**: `(1, 3, 512)` — same as input

---

## Step 8: Stack N Layers

The Transformer block (attention + FFN) is repeated N times:

```
Input embeddings
    ↓
┌─────────────────┐
│ Transformer     │
│ Block 1         │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Transformer     │
│ Block 2         │
└────────┬────────┘
         ↓
        ...
         ↓
┌─────────────────┐
│ Transformer     │
│ Block N         │
└────────┬────────┘
         ↓
Final representations
```

**Typical values**:
- GPT-2 Small: 12 layers
- GPT-2 Medium: 24 layers
- GPT-3: 96 layers
- GPT-4: Unknown (estimated 120+)

Each layer refines the representations, building more abstract features.

---

## Step 9: Output Layer

### For language modeling (GPT)

Convert final representations to vocabulary probabilities:

```python
# Final hidden states: (batch, seq_len, d_model)
# Output projection: (d_model, vocab_size)

logits = hidden @ W_out  # (batch, seq_len, vocab_size)
probs = softmax(logits, dim=-1)
```

### Weight tying

Many models share the embedding matrix with the output projection:

```python
W_out = embedding_table.T  # Transpose of input embeddings
```

**Why?** Reduces parameters and improves performance.

### Generating text

```python
# Get probability distribution for next token
logits = model(input_ids)[:, -1, :]  # Last position
probs = softmax(logits)

# Sample or take argmax
next_token = torch.multinomial(probs, 1)  # Sample
# or
next_token = torch.argmax(probs)  # Greedy
```

---

## Complete Forward Pass: Putting It All Together

```python
def transformer_forward(token_ids):
    # 1. Token embedding
    x = embedding_table(token_ids)  # (batch, seq_len, d_model)
    
    # 2. Add positional encoding
    x = x + positional_encoding(seq_len)
    
    # 3. Transformer blocks
    for layer in transformer_layers:
        # Self-attention with residual
        attn_out = layer.attention(layer.norm1(x))
        x = x + attn_out
        
        # FFN with residual
        ffn_out = layer.ffn(layer.norm2(x))
        x = x + ffn_out
    
    # 4. Final layer norm
    x = final_norm(x)
    
    # 5. Output projection
    logits = x @ output_projection  # (batch, seq_len, vocab_size)
    
    return logits
```

---

## Encoder vs Decoder vs Encoder-Decoder

### Encoder-only (BERT)

```
Input: "The cat [MASK] on the mat"
       ↓
   Bidirectional self-attention (see all tokens)
       ↓
Output: Representations for each token
```

**Use cases**: Classification, NER, sentence embeddings

### Decoder-only (GPT)

```
Input: "The cat sat"
       ↓
   Causal self-attention (only see past tokens)
       ↓
Output: Predict next token "on"
```

**Use cases**: Text generation, chatbots, code completion

### Encoder-Decoder (T5, BART)

```
Encoder input: "Translate to French: The cat sat"
       ↓
   Bidirectional self-attention
       ↓
Encoder output: Context representations
       ↓
Decoder input: "<start>"
       ↓
   Causal self-attention + Cross-attention to encoder
       ↓
Decoder output: "Le chat était assis"
```

**Use cases**: Translation, summarization, question answering

---

## Parameter Count

Let's count parameters for a GPT-2 Small (d_model=768, n_layers=12, n_heads=12):

| Component | Parameters | Formula |
|-----------|------------|---------|
| Token embeddings | 38.6M | vocab_size × d_model = 50257 × 768 |
| Position embeddings | 0.8M | max_seq_len × d_model = 1024 × 768 |
| Per-layer attention | 2.4M | 4 × d_model² = 4 × 768² |
| Per-layer FFN | 4.7M | 2 × d_model × 4×d_model = 2 × 768 × 3072 |
| Per-layer norms | 3K | 4 × d_model = 4 × 768 |
| **Total per layer** | **7.1M** | |
| **12 layers** | **85M** | |
| Output projection | 0 | (tied with embeddings) |
| **Grand total** | **~124M** | |

---

## Computational Complexity

### Self-attention

- **Time**: O(n² × d) where n = sequence length, d = d_model
- **Memory**: O(n²) for attention weights

This is why long sequences are expensive!

```
Sequence length    Attention operations
128               16,384
512               262,144
2048              4,194,304
8192              67,108,864
```

### Solutions for long sequences

| Method | Complexity | Used in |
|--------|------------|---------|
| Sparse attention | O(n√n) | Longformer, BigBird |
| Linear attention | O(n) | Performer, Linear Transformer |
| Sliding window | O(n × w) | Mistral |
| Flash Attention | O(n²) but faster | Most modern models |

---

## Training vs Inference

### Training (parallel)

All tokens processed simultaneously:

```python
# Input: "The cat sat on the mat"
# Target: "cat sat on the mat <eos>"

logits = model(input_ids)  # (batch, seq_len, vocab)
loss = cross_entropy(logits, target_ids)
```

The causal mask ensures each position only attends to previous positions, but computation is parallel.

### Inference (sequential)

Generate one token at a time:

```python
generated = [start_token]
for _ in range(max_length):
    logits = model(generated)
    next_token = sample(logits[:, -1, :])
    generated.append(next_token)
    if next_token == eos_token:
        break
```

**KV-Cache optimization**: Store computed K, V from previous tokens to avoid recomputation.

---

## Summary: The Complete Picture

```
"The cat sat"
      ↓
┌─────────────────────────────────────────────────────────┐
│ 1. TOKENIZE: ["The", " cat", " sat"] → [464, 3797, 3332]│
└─────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────┐
│ 2. EMBED: Look up 512-dim vectors for each token        │
└─────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────┐
│ 3. ADD POSITIONS: Add sinusoidal or learned positions   │
└─────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────┐
│ 4. TRANSFORMER BLOCK (×N)                               │
│    ├── Multi-Head Self-Attention                        │
│    │   └── Q, K, V projections → Attention → Concat     │
│    ├── Add & LayerNorm                                  │
│    ├── Feed-Forward Network (expand → activate → contract)│
│    └── Add & LayerNorm                                  │
└─────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────┐
│ 5. OUTPUT: Project to vocab size → softmax → next token │
└─────────────────────────────────────────────────────────┘
      ↓
"on" (predicted next token)
```

---

## Interview Questions

### Conceptual Questions

**Q1: Walk me through what happens when a Transformer processes the sentence "Hello world".**

<details>
<summary>Answer</summary>

1. **Tokenization**: "Hello world" → token IDs (e.g., [15496, 995])
2. **Embedding**: Look up 512-dim vectors for each token
3. **Positional encoding**: Add position information (pos 0, pos 1)
4. **Self-attention**: Each token attends to all tokens, computing weighted sum of values
5. **FFN**: Each token passes through a 2-layer MLP
6. **Repeat**: Stack N transformer blocks
7. **Output**: Project final representations to vocabulary, get probability distribution for next token
</details>

**Q2: Why do we need positional encoding? What would happen without it?**

<details>
<summary>Answer</summary>

Self-attention is permutation-invariant — it computes the same output regardless of token order. Without positional encoding:
- "dog bites man" and "man bites dog" would produce identical representations
- The model couldn't distinguish word order
- Language understanding would be impossible

Positional encoding adds unique position information to each token's embedding.
</details>

**Q3: Explain the Query, Key, Value mechanism. Why three separate projections?**

<details>
<summary>Answer</summary>

- **Query (Q)**: "What am I looking for?" — represents the current token's information need
- **Key (K)**: "What do I contain?" — represents what each token offers
- **Value (V)**: "What information do I provide?" — the actual content to aggregate

Three projections allow the model to:
1. Learn different representations for matching (Q, K) vs content (V)
2. Have asymmetric relationships (token A might attend to B, but not vice versa)
3. Separate "relevance" from "content"

Analogy: In a library, Q is your search query, K is book titles, V is book contents.
</details>

**Q4: Why do we scale attention scores by √d_k?**

<details>
<summary>Answer</summary>

For high-dimensional vectors (e.g., d_k = 64):
- Dot products can become very large (variance grows with dimension)
- Large values cause softmax to saturate (outputs near 0 or 1)
- Saturated softmax has near-zero gradients → training fails

Scaling by √d_k keeps the variance of dot products around 1, preventing saturation.

Mathematical intuition: If Q and K have unit variance, then QK^T has variance d_k. Dividing by √d_k normalizes this back to variance 1.
</details>

**Q5: What's the difference between encoder-only, decoder-only, and encoder-decoder Transformers?**

<details>
<summary>Answer</summary>

| Type | Attention | Use Case | Examples |
|------|-----------|----------|----------|
| **Encoder-only** | Bidirectional (see all tokens) | Understanding tasks | BERT, RoBERTa |
| **Decoder-only** | Causal (only see past) | Generation | GPT, LLaMA |
| **Encoder-Decoder** | Encoder: bidirectional, Decoder: causal + cross-attention | Seq2seq | T5, BART |

Key insight: Decoder-only models can do everything with enough scale (GPT-4), but encoder-decoder is more efficient for translation/summarization.
</details>

### Technical Questions

**Q6: What is the computational complexity of self-attention? Why is this a problem?**

<details>
<summary>Answer</summary>

- **Time complexity**: O(n² × d) where n = sequence length
- **Memory complexity**: O(n²) for storing attention weights

Problem: Quadratic scaling with sequence length
- 1K tokens: 1M operations
- 10K tokens: 100M operations
- 100K tokens: 10B operations

Solutions: Sparse attention, linear attention, sliding window, Flash Attention
</details>

**Q7: Calculate the number of parameters in a single Transformer layer with d_model=512, n_heads=8, d_ff=2048.**

<details>
<summary>Answer</summary>

**Attention**:
- W_Q, W_K, W_V, W_O: 4 × (512 × 512) = 1,048,576
- Biases: 4 × 512 = 2,048

**FFN**:
- W1: 512 × 2048 = 1,048,576
- b1: 2048
- W2: 2048 × 512 = 1,048,576
- b2: 512

**Layer norms** (2 per layer):
- gamma, beta: 2 × 2 × 512 = 2,048

**Total**: ~3.15M parameters per layer
</details>

**Q8: What is the KV-cache and why is it important for inference?**

<details>
<summary>Answer</summary>

During autoregressive generation, we generate one token at a time. Without KV-cache:
- To generate token 100, we'd recompute attention for all 99 previous tokens
- O(n²) computation for each new token

With KV-cache:
- Store K and V matrices from previous tokens
- Only compute Q for the new token
- Reduces per-token computation from O(n²) to O(n)

Trade-off: Uses more memory (storing all K, V), but much faster inference.
</details>

**Q9: Explain the difference between pre-norm and post-norm. Which is better?**

<details>
<summary>Answer</summary>

**Post-norm** (original Transformer):
```
output = LayerNorm(x + Sublayer(x))
```

**Pre-norm** (GPT-2, modern):
```
output = x + Sublayer(LayerNorm(x))
```

Pre-norm is generally better because:
1. Gradients flow more directly through residual connections
2. More stable training, especially for deep networks
3. Often doesn't require learning rate warmup

Post-norm can achieve slightly better final performance but is harder to train.
</details>

**Q10: Why do we use Layer Normalization instead of Batch Normalization in Transformers?**

<details>
<summary>Answer</summary>

**Batch Norm**: Normalizes across the batch dimension
- Depends on batch statistics (mean, variance of other samples)
- Problematic for variable-length sequences
- Different behavior at train vs inference time

**Layer Norm**: Normalizes across the feature dimension
- Independent per sample
- Works with any batch size (even 1)
- Same behavior at train and inference

For sequential data with variable lengths, Layer Norm is more stable and consistent.
</details>

### Coding Questions

**Q11: Implement scaled dot-product attention from scratch.**

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, n_heads, seq_len, d_k)
    K: (batch, n_heads, seq_len, d_k)
    V: (batch, n_heads, seq_len, d_v)
    mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
    """
    # Your code here
    pass
```

<details>
<summary>Solution</summary>

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask (for causal attention or padding)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```
</details>

**Q12: Implement a causal mask for a sequence of length n.**

<details>
<summary>Solution</summary>

```python
def create_causal_mask(seq_len):
    """
    Creates a lower triangular mask.
    Returns: (1, 1, seq_len, seq_len) tensor
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)

# Example for seq_len=4:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```
</details>

**Q13: Implement the feed-forward network with GELU activation.**

<details>
<summary>Solution</summary>

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.linear1(x)           # (batch, seq_len, d_ff)
        x = F.gelu(x)                 # GELU activation
        x = self.dropout(x)
        x = self.linear2(x)           # (batch, seq_len, d_model)
        return x
```
</details>

### System Design Questions

**Q14: You need to deploy a Transformer model for real-time chat. What optimizations would you consider?**

<details>
<summary>Answer</summary>

1. **KV-Cache**: Store key/value pairs to avoid recomputation
2. **Quantization**: INT8 or INT4 weights (reduce memory, faster compute)
3. **Batching**: Dynamic batching for multiple concurrent requests
4. **Speculative decoding**: Use small model to draft, large model to verify
5. **Flash Attention**: Memory-efficient attention implementation
6. **Model parallelism**: Split model across GPUs for large models
7. **Pruning**: Remove unnecessary attention heads or layers
8. **Distillation**: Train smaller model to mimic larger one
</details>

**Q15: How would you modify a Transformer to handle 100K token sequences efficiently?**

<details>
<summary>Answer</summary>

Standard attention is O(n²), so 100K tokens = 10B operations per layer.

Solutions:
1. **Sliding window attention** (Mistral): Only attend to nearby tokens
2. **Sparse attention** (Longformer): Local + global attention patterns
3. **Linear attention** (Performer): Approximate softmax with kernels
4. **Hierarchical**: Summarize chunks, then attend to summaries
5. **Retrieval-augmented**: Store long context externally, retrieve relevant parts
6. **RoPE + ALiBi**: Better position encodings that generalize to longer sequences
</details>

**Q16: Explain how you would implement multi-head attention efficiently on a GPU.**

<details>
<summary>Answer</summary>

Key insight: Treat heads as a batch dimension for parallelism.

```python
# Instead of looping over heads:
# for i in range(n_heads):
#     head_i = attention(Q_i, K_i, V_i)

# Do all heads in parallel:
# 1. Project Q, K, V for all heads at once
Q = x @ W_Q  # (batch, seq_len, n_heads * d_k)

# 2. Reshape to separate heads
Q = Q.view(batch, seq_len, n_heads, d_k).transpose(1, 2)
# Now: (batch, n_heads, seq_len, d_k)

# 3. Attention treats n_heads as batch dimension
# Single batched matrix multiply handles all heads

# 4. Reshape back and project
output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
output = output @ W_O
```

This leverages GPU parallelism across both batch and head dimensions.
</details>
