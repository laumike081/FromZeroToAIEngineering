# NLP Foundations: From Word Vectors to Transformers

This chapter provides a deeper conceptual understanding of the key architectures that led to modern LLMs. We cover the "why" behind each innovation, not just the "what."

---

## Overview: The Path to Transformers

```
Statistical NLP → Word Embeddings → RNNs → Attention → Transformers
   (1990s)          (2013)         (2014)   (2015)      (2017)
```

Each step solved a limitation of the previous approach:

| Era | Problem Solved | New Limitation |
|-----|----------------|----------------|
| **Word Embeddings** | Words as discrete symbols → dense vectors | Static (one embedding per word) |
| **RNNs** | No sequence modeling → hidden state memory | Sequential, slow, vanishing gradients |
| **LSTM/GRU** | Vanishing gradients → gated memory | Still sequential, limited parallelism |
| **Attention** | Fixed-size bottleneck → dynamic focus | Added to RNNs, not replacing them |
| **Transformers** | Sequential processing → full parallelism | Quadratic attention cost |

---

## Part 1: Tokenization

Before any model can process text, we need to convert it to numbers. This seemingly simple step has profound implications for model performance.

### Why Tokenization Matters

```
"I love machine learning" → [1, 42, 156, 89] → Model → [0.2, 0.8, ...] → "positive"
```

Tokenization determines:
- **Vocabulary size**: How many unique tokens the model knows
- **Sequence length**: How many tokens represent a given text
- **Semantic granularity**: Whether meaning is captured at word, subword, or character level
- **Out-of-vocabulary handling**: What happens with unseen words

### Tokenization Strategies

| Method | Example | Pros | Cons |
|--------|---------|------|------|
| **Word-level** | `["I", "love", "ML"]` | Intuitive, semantic | Huge vocab, OOV problem |
| **Character-level** | `["I", " ", "l", "o", "v", "e"]` | Small vocab, no OOV | Very long sequences, loses word meaning |
| **Subword (BPE)** | `["I", "love", "machine", "learn", "ing"]` | Balance of both | Requires training tokenizer |

#### Detailed Example: Word-Level Tokenization

```python
text = "The cat sat on the mat"
tokens = text.split()  # ["The", "cat", "sat", "on", "the", "mat"]

# Problem: What about "ChatGPT" or "don't" or "café"?
text2 = "I can't believe ChatGPT works!"
tokens2 = text2.split()  # ["I", "can't", "believe", "ChatGPT", "works!"]
# "can't" is one token, "works!" includes punctuation - inconsistent!
```

**The OOV (Out-of-Vocabulary) Problem**:
- Training vocabulary: 50,000 words
- User types: "transformerify" (not in vocab)
- Model sees: `<UNK>` (unknown token) → loses all meaning!

#### Detailed Example: Character-Level Tokenization

```python
text = "Hello"
tokens = list(text)  # ["H", "e", "l", "l", "o"]

# Pros: Vocabulary is just ~100 characters (letters, digits, punctuation)
# Cons: "Hello world" = 11 tokens instead of 2
#       Model must learn that "H"+"e"+"l"+"l"+"o" = greeting
```

### Byte-Pair Encoding (BPE)

Modern LLMs use **subword tokenization** (BPE, WordPiece, SentencePiece). BPE is a compression algorithm repurposed for NLP.

**Algorithm**:

1. Start with character vocabulary
2. Count all adjacent character pairs in corpus
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary size reached

**Step-by-step example**:

```
Corpus: "low lower lowest"

Step 0 - Character vocab: {l, o, w, e, r, s, t, _}
         Tokens: ["l", "o", "w", "_"], ["l", "o", "w", "e", "r", "_"], ...

Step 1 - Most frequent pair: ("l", "o") appears 3 times
         Merge: "lo" becomes a token
         Tokens: ["lo", "w", "_"], ["lo", "w", "e", "r", "_"], ...

Step 2 - Most frequent pair: ("lo", "w") appears 3 times
         Merge: "low" becomes a token
         Tokens: ["low", "_"], ["low", "e", "r", "_"], ["low", "e", "s", "t", "_"]

Step 3 - Most frequent pair: ("e", "r") appears 1 time, ("e", "s") appears 1 time
         ... continue until vocab size reached
```

**Real-world tokenization examples** (GPT-4 tokenizer):

```
"Hello world"     → ["Hello", " world"]           (2 tokens)
"Transformers"    → ["Transform", "ers"]          (2 tokens)
"ChatGPT"         → ["Chat", "G", "PT"]           (3 tokens)
"🎉"              → ["\xf0\x9f", "\x8e\x89"]      (2 tokens - emoji as bytes)
"café"            → ["caf", "é"]                  (2 tokens)
"antidisestablish" → ["anti", "dis", "establish"] (3 tokens)
```

**Key insight**: Common words stay whole, rare words split into subwords. This handles:
- **Morphology**: "running" → ["run", "ning"] - model learns "ning" = present participle
- **New words**: "ChatGPT" → ["Chat", "G", "PT"] - combines known subwords
- **Typos**: "teh" → ["t", "eh"] - graceful degradation
- **Languages**: Works across languages without language-specific rules

### WordPiece vs BPE vs SentencePiece

| Algorithm | Used By | Key Difference |
|-----------|---------|----------------|
| **BPE** | GPT-2, GPT-3, GPT-4 | Merges most frequent pairs |
| **WordPiece** | BERT, DistilBERT | Merges pairs that maximize likelihood |
| **SentencePiece** | T5, LLaMA | Language-agnostic, treats text as bytes |
| **Tiktoken** | OpenAI models | Optimized BPE implementation |

---

## Part 2: Word Representation

### The Problem with One-Hot Encoding

Traditional NLP represented words as one-hot vectors:

```
Vocabulary: [cat, dog, king, queen, man, woman]

cat   = [1, 0, 0, 0, 0, 0]
dog   = [0, 1, 0, 0, 0, 0]
king  = [0, 0, 1, 0, 0, 0]
queen = [0, 0, 0, 1, 0, 0]
```

**Problems**:
- **Sparse**: Vectors are mostly zeros
- **No similarity**: `cosine(cat, dog) = 0` (orthogonal)
- **Huge**: Vocabulary of 50K words = 50K-dimensional vectors

### Word2Vec: Learning Dense Representations

**Key insight**: "You shall know a word by the company it keeps" (Firth, 1957)

Words that appear in similar contexts should have similar embeddings. This is the **distributional hypothesis**.

**Example**: Consider these sentences:
- "I adopted a **cat** from the shelter"
- "I adopted a **dog** from the shelter"
- "I adopted a **kitten** from the shelter"

"Cat", "dog", and "kitten" appear in identical contexts, so they should have similar embeddings.

#### The Architecture

Word2Vec is a shallow neural network with:
- **Input layer**: One-hot encoded word (vocab_size dimensions)
- **Hidden layer**: Dense embedding (e.g., 300 dimensions) - NO activation function!
- **Output layer**: Probability distribution over vocabulary

```
Input (V dims)     Hidden (D dims)     Output (V dims)
    [0]                                    [0.01]
    [0]              [0.2]                 [0.02]
    [1]  ----W1---->  [0.5]  ----W2---->   [0.85]  <- "sat"
    [0]              [-0.1]                [0.05]
    [0]              [0.3]                 [0.07]
   "cat"           embedding              softmax
```

The **hidden layer weights W1** become our word embeddings!

#### Two Training Objectives

**1. CBOW (Continuous Bag of Words)**

Predict the center word from context words:

```
Sentence: "The cat sat on the mat"
Window size: 2

Context: ["The", "cat", "on", "the"]  →  Target: "sat"
         (2 words left, 2 words right)

Input: average([emb(The), emb(cat), emb(on), emb(the)]) → Model → Output: sat
```

**CBOW characteristics**:
- Faster to train (one prediction per window)
- Better for frequent words (more training signal)
- Smooths over context (averages embeddings)
- Works well with large datasets

**2. Skip-gram**

Predict context words from the center word:

```
Sentence: "The cat sat on the mat"
Window size: 2
Center: "sat"

Training pairs generated:
  (sat, The)   - predict "The" from "sat"
  (sat, cat)   - predict "cat" from "sat"
  (sat, on)    - predict "on" from "sat"
  (sat, the)   - predict "the" from "sat"
```

**Skip-gram characteristics**:
- Slower (multiple predictions per center word)
- Better for rare words (each occurrence generates multiple training examples)
- Captures more nuanced relationships
- Works well with small datasets

#### The Magic of Word Vectors

After training, word vectors capture semantic AND syntactic relationships:

**Semantic relationships** (meaning):
```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
vector("Paris") - vector("France") + vector("Italy") ≈ vector("Rome")
vector("walking") - vector("walked") + vector("swam") ≈ vector("swimming")
```

**Why does this work?**

The model learns that:
- "king" and "queen" appear in similar royal contexts
- "man" and "woman" differ along a gender dimension
- The gender dimension is consistent across word pairs

```
        man ----gender----> woman
         |                    |
      royalty              royalty
         |                    |
        king ---gender----> queen
```

**Syntactic relationships** (grammar):
```
vector("biggest") - vector("big") + vector("small") ≈ vector("smallest")
vector("running") - vector("run") + vector("swim") ≈ vector("swimming")
```

#### Training with Negative Sampling

**The problem**: Full softmax over vocabulary is expensive.

For vocabulary V = 50,000 words:
- Output layer: 50,000 neurons
- Each training step: compute 50,000 probabilities
- Very slow!

**The solution**: **Negative sampling** approximates softmax.

Instead of predicting the exact word, train a binary classifier:
- **Positive example**: (center, context) pairs that actually occur
- **Negative examples**: (center, random_word) pairs that don't occur

```python
# Training example: "The cat sat on the mat"
# Center: "sat", Context: "cat"

# Positive pair: (sat, cat) - these words DO appear together
loss += -log(sigmoid(dot(emb_sat, emb_cat)))
# Push embeddings closer: sigmoid(dot) should be close to 1

# Negative pairs: (sat, quantum), (sat, elephant), (sat, democracy)
# These words DON'T appear together
for neg_word in ["quantum", "elephant", "democracy"]:
    loss += -log(sigmoid(-dot(emb_sat, emb_neg_word)))
    # Push embeddings apart: sigmoid(-dot) should be close to 1
```

**How many negative samples?**
- Small datasets: 5-20 negative samples
- Large datasets: 2-5 negative samples
- Original paper: 5 for large datasets, 15 for small

**How to sample negative words?**
- Sample proportional to word frequency raised to 3/4 power
- `P(w) ∝ frequency(w)^0.75`
- This downweights very common words ("the", "a") and upweights medium-frequency words

### GloVe: An Alternative Approach

**GloVe** (Global Vectors) takes a different approach:

1. Build a co-occurrence matrix: count how often words appear together
2. Factorize the matrix to get embeddings

```
Co-occurrence matrix (simplified):
        cat   dog   sat   mat
cat     -     5     10    8
dog     5     -     12    6
sat     10    12    -     15
mat     8     6     15    -
```

**GloVe objective**: Make dot product of embeddings equal log of co-occurrence:
```
dot(emb_i, emb_j) ≈ log(X_ij)
```

**Word2Vec vs GloVe**:
- Word2Vec: Local context windows, online training
- GloVe: Global statistics, batch training
- In practice: Similar performance, Word2Vec more popular

### Limitations of Word2Vec

1. **Static embeddings**: One vector per word, regardless of context
   ```
   "I deposited money at the bank"  → bank = [0.2, 0.5, ...]
   "I sat by the river bank"        → bank = [0.2, 0.5, ...]  (same!)
   ```

2. **No sentence understanding**: Bag of words, order doesn't matter
   ```
   "dog bites man" vs "man bites dog"  → same word embeddings!
   ```

3. **Out-of-vocabulary**: Can't handle new words
   ```
   "transformerify" → not in vocabulary → no embedding
   ```

4. **No compositionality**: Can't build phrase meanings from word meanings
   ```
   "hot dog" ≠ hot + dog
   "kick the bucket" ≠ kick + the + bucket
   ```

These limitations led to the development of **contextual embeddings** (ELMo, BERT) and **sequence models** (RNNs, Transformers).

---

## Part 3: Recurrent Neural Networks (RNNs)

### The Idea: Memory Through Hidden State

RNNs process sequences one element at a time, maintaining a "hidden state" that summarizes what came before. This is fundamentally different from feedforward networks that process fixed-size inputs.

**Why do we need sequence models?**

Consider these tasks:
- **Sentiment analysis**: "This movie was not good" vs "This movie was good" (word order matters!)
- **Translation**: "I ate an apple" → "J'ai mangé une pomme" (variable length input/output)
- **Language modeling**: "The cat sat on the ___" (predict based on history)

Word2Vec can't handle these because it ignores word order.

### RNN Architecture

```
Input:  x₁    x₂    x₃    x₄
         ↓     ↓     ↓     ↓
State:  h₀ → h₁ → h₂ → h₃ → h₄
         ↓     ↓     ↓     ↓
Output: y₁    y₂    y₃    y₄
```

**Key insight**: The same weights are used at every timestep (weight sharing). This means:
- The network can handle sequences of any length
- It learns general patterns, not position-specific ones

### The Math

At each timestep t:

```
hₜ = tanh(Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + bₕ)
yₜ = Wₕᵧ · hₜ + bᵧ
```

Where:
- `xₜ`: Input at time t (e.g., word embedding)
- `hₜ`: Hidden state at time t (the "memory")
- `Wₓₕ`: Input-to-hidden weights (how to incorporate new input)
- `Wₕₕ`: Hidden-to-hidden weights (how to transform memory)
- `Wₕᵧ`: Hidden-to-output weights (how to produce output)

**Concrete example**:

```python
# Processing "The cat sat"
# Assume embedding_dim = 4, hidden_dim = 3

h0 = [0, 0, 0]  # Initial hidden state (zeros)

# Step 1: Process "The"
x1 = [0.1, 0.2, 0.3, 0.4]  # embedding of "The"
h1 = tanh(Wxh @ x1 + Whh @ h0 + bh)
# h1 = [0.2, -0.1, 0.5]  # Now contains info about "The"

# Step 2: Process "cat"
x2 = [0.5, 0.1, 0.2, 0.3]  # embedding of "cat"
h2 = tanh(Wxh @ x2 + Whh @ h1 + bh)
# h2 = [0.4, 0.3, 0.1]  # Now contains info about "The cat"

# Step 3: Process "sat"
x3 = [0.3, 0.4, 0.1, 0.2]  # embedding of "sat"
h3 = tanh(Wxh @ x3 + Whh @ h2 + bh)
# h3 = [0.1, 0.6, 0.2]  # Now contains info about "The cat sat"

# Final hidden state h3 summarizes the entire sequence!
```

### RNN Variants

**Many-to-One** (e.g., sentiment classification):
```
Input:  x₁    x₂    x₃    x₄
         ↓     ↓     ↓     ↓
State:  h₀ → h₁ → h₂ → h₃ → h₄
                              ↓
Output:                      y  (single output from final state)
```

**One-to-Many** (e.g., image captioning):
```
Input:  x
         ↓
State:  h₀ → h₁ → h₂ → h₃
         ↓     ↓     ↓     ↓
Output: y₁    y₂    y₃    y₄
```

**Many-to-Many** (e.g., translation):
```
Encoder:  x₁ → x₂ → x₃ → [context]
Decoder:  [context] → y₁ → y₂ → y₃
```

### The Vanishing Gradient Problem

During backpropagation, gradients flow backward through time (BPTT - Backpropagation Through Time):

```
∂L/∂h₁ = ∂L/∂h₄ · ∂h₄/∂h₃ · ∂h₃/∂h₂ · ∂h₂/∂h₁
```

Each `∂hₜ/∂hₜ₋₁` involves multiplying by `Wₕₕ` and the derivative of tanh.

**The derivative of tanh**:
```
tanh'(x) = 1 - tanh²(x)
```
- Maximum value: 1 (when x = 0)
- For most values: < 1
- Saturates to 0 for large |x|

**Problem**: If these values are < 1, the gradient shrinks exponentially:
```
Gradient after 100 steps: 0.9¹⁰⁰ ≈ 0.00003
Gradient essentially disappears!
```

**Concrete example**:
```
Sentence: "The cat, which was sitting on the mat in the living room, was happy."

To learn that "was" agrees with "cat" (not "room"), the gradient must flow:
was → room → living → the → in → mat → the → on → sitting → was → which → cat

That's 11 steps! Gradient is nearly zero by the time it reaches "cat".
```

**Result**: RNNs struggle to learn long-range dependencies.

### The Exploding Gradient Problem

The opposite can also happen: if eigenvalues of Wₕₕ > 1, gradients explode:
```
Gradient after 100 steps: 1.1¹⁰⁰ ≈ 13,780
Gradient becomes NaN!
```

**Solution**: Gradient clipping
```python
if gradient_norm > threshold:
    gradient = gradient * (threshold / gradient_norm)
```

This prevents explosion but doesn't help vanishing gradients.

---

## Part 4: LSTM (Long Short-Term Memory)

### The Solution: Gated Memory

LSTMs (Hochreiter & Schmidhuber, 1997) add a **cell state** (long-term memory) and **gates** to control information flow. The key innovation is creating a "highway" for gradients to flow through time.

```
        ┌─────────────────────────────────────┐
        │           Cell State Cₜ             │
        │  ←forget→  ←──add──→  ←──output──→  │
        └─────────────────────────────────────┘
              ↑           ↑           ↓
           ┌──┴──┐     ┌──┴──┐     ┌──┴──┐
           │  fₜ │     │  iₜ │     │  oₜ │
           │Forget│    │Input│     │Output│
           │ Gate │    │Gate │     │ Gate │
           └──┬──┘     └──┬──┘     └──┬──┘
              └─────┬─────┴─────┬─────┘
                    │           │
                   hₜ₋₁        xₜ
```

### The Intuition: A Conveyor Belt

Think of the cell state as a **conveyor belt** running through time:
- Information can be placed on the belt (input gate)
- Information can be removed from the belt (forget gate)
- Information can be read from the belt (output gate)

The belt itself just moves forward with minimal transformation, allowing information to travel long distances.

### The Three Gates

All gates use sigmoid activation (σ), outputting values between 0 and 1:
- 0 = "completely block"
- 1 = "completely allow"
- 0.5 = "allow half"

**1. Forget Gate** - What to remove from memory

```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
```

**Example**: Processing "The cat sat. The dog ran."
- When we see "The dog", the forget gate might output low values for "cat"-related information
- This clears space for new subject information

**2. Input Gate** - What new information to add

```
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)      # How much to write
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)   # What to write (candidate values)
```

**Example**: Processing "The cat sat"
- Input gate decides: "sat" is important, write it strongly
- Candidate: creates a representation of "sitting action"

**3. Output Gate** - What to output from memory

```
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
hₜ = oₜ * tanh(Cₜ)
```

**Example**: For next-word prediction after "The cat sat on the"
- Output gate might emphasize location-related information
- Suppresses irrelevant stored information

### Cell State Update

```
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ
     └────────┘   └───────┘
      keep old    add new
```

**Key insight**: The cell state update is just element-wise multiplication and addition!
- No matrix multiplication (unlike RNN's hₜ = tanh(Wₕₕ · hₜ₋₁ + ...))
- Gradients flow directly through the + operation
- If fₜ ≈ 1 (don't forget), gradient flows unchanged

**Concrete example**:

```python
# Processing: "The cat sat on the mat"
# Cell state tracks: subject, action, location

# After "The cat":
C1 = [0.8, 0.0, 0.0]  # Strong subject signal, no action/location yet

# After "sat":
f2 = [1.0, 0.5, 0.5]  # Keep subject, partially forget others
i2 = [0.1, 0.9, 0.1]  # Write action strongly
C_candidate = [0.1, 0.7, 0.0]  # New candidate
C2 = f2 * C1 + i2 * C_candidate
   = [0.8, 0.0, 0.0] + [0.01, 0.63, 0.0]
   = [0.81, 0.63, 0.0]  # Now has subject AND action

# After "on the mat":
C5 = [0.75, 0.55, 0.8]  # Subject, action, AND location
```

### Why LSTM Solves Vanishing Gradients

**RNN gradient path**:
```
∂Cₜ/∂C₁ = Wₕₕ · Wₕₕ · Wₕₕ · ... (t-1 matrix multiplications)
```

**LSTM gradient path**:
```
∂Cₜ/∂C₁ = fₜ · fₜ₋₁ · fₜ₋₂ · ... (element-wise multiplications)
```

If forget gates are close to 1, gradients flow almost unchanged!

### GRU: A Simpler Alternative

**Gated Recurrent Unit** (Cho et al., 2014) combines forget and input gates:

```
Update gate: zₜ = σ(Wz · [hₜ₋₁, xₜ])   # How much to update
Reset gate:  rₜ = σ(Wr · [hₜ₋₁, xₜ])   # How much of past to forget
Candidate:   h̃ₜ = tanh(W · [rₜ * hₜ₋₁, xₜ])
Output:      hₜ = (1 - zₜ) * hₜ₋₁ + zₜ * h̃ₜ
```

**Key difference**: GRU has no separate cell state. The hidden state serves both purposes.

**LSTM vs GRU**:

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| States | 2 (cell state, hidden state) | 1 (hidden state) |
| Parameters | More | Fewer (~25% less) |
| Performance | Slightly better on long sequences | Similar, faster to train |
| Use case | When you need fine-grained control | Default choice, simpler |

### Bidirectional RNNs

Sometimes context from the future matters:
- "I read the book" vs "I will read the book" - "read" pronunciation depends on tense
- Named entity recognition: "Apple" (company) vs "apple" (fruit) depends on surrounding words

**Solution**: Run two RNNs, one forward and one backward:

```
Forward:   h→₁ → h→₂ → h→₃ → h→₄
Backward:  h←₁ ← h←₂ ← h←₃ ← h←₄
Combined:  [h→₁;h←₁] [h→₂;h←₂] [h→₃;h←₃] [h→₄;h←₄]
```

Each position now has context from both past AND future.

### Remaining Limitations

Even with LSTM/GRU:
- **Still sequential**: Can't parallelize across time (must compute h₁ before h₂)
- **Fixed hidden size**: All information compressed into one vector (bottleneck)
- **Practical limit**: ~100-500 tokens before degradation
- **Training time**: O(sequence_length) - can't use GPU parallelism effectively

---

## Part 5: The Attention Mechanism

### The Bottleneck Problem

In sequence-to-sequence models (e.g., translation), the encoder compresses the entire input into a single vector:

```
"The cat sat on the mat" → [0.2, -0.5, 0.8, ...] → "Le chat..."
                              ↑
                    Single fixed-size vector!
```

**The problem**: How do you compress "The quick brown fox jumps over the lazy dog" into 256 numbers? What about a 1000-word paragraph?

- Short sentences: works okay
- Long sentences: information loss, poor translation quality
- The last hidden state is biased toward recent words

### Attention: Dynamic Focus

**Key insight** (Bahdanau et al., 2015): Instead of one summary vector, let the decoder look at ALL encoder states and decide which are relevant for each output word.

```
Encoder states: [h₁, h₂, h₃, h₄, h₅]  (one per input word)
                  ↓   ↓   ↓   ↓   ↓
Attention:      [0.1, 0.1, 0.6, 0.1, 0.1]  (weights sum to 1)
                  ↓   ↓   ↓   ↓   ↓
Context:        weighted sum of encoder states
```

**Example**: Translating "The cat sat on the mat" → "Le chat était assis sur le tapis"

```
Generating "chat" (cat):
  Attention: [0.05, 0.85, 0.05, 0.02, 0.02, 0.01]
                    ^^^^
             Focus on "cat"!

Generating "tapis" (mat):
  Attention: [0.02, 0.02, 0.03, 0.03, 0.05, 0.85]
                                           ^^^^
                                    Focus on "mat"!
```

The model learns to align source and target words automatically!

### Computing Attention Scores

For each decoder state `sₜ`, compute attention over encoder states `h₁...hₙ`:

```
1. Score:    eₜᵢ = score(sₜ, hᵢ)           # How relevant is hᵢ?
2. Normalize: αₜᵢ = softmax(eₜᵢ)           # Convert to probabilities
3. Context:  cₜ = Σᵢ αₜᵢ · hᵢ              # Weighted sum
```

**Step-by-step example**:

```python
# Encoder states for "The cat sat"
h1 = [0.1, 0.2, 0.3]  # "The"
h2 = [0.8, 0.1, 0.4]  # "cat"
h3 = [0.2, 0.7, 0.1]  # "sat"

# Decoder state (trying to generate "chat")
s = [0.7, 0.2, 0.5]

# Step 1: Compute scores (using dot product)
e1 = dot(s, h1) = 0.1*0.7 + 0.2*0.2 + 0.3*0.5 = 0.26
e2 = dot(s, h2) = 0.8*0.7 + 0.1*0.2 + 0.4*0.5 = 0.78  # Highest!
e3 = dot(s, h3) = 0.2*0.7 + 0.7*0.2 + 0.1*0.5 = 0.33

# Step 2: Normalize with softmax
α = softmax([0.26, 0.78, 0.33]) = [0.24, 0.47, 0.29]
                                        ^^^^
                                  "cat" gets most attention

# Step 3: Weighted sum
context = 0.24*h1 + 0.47*h2 + 0.29*h3
        = [0.024, 0.048, 0.072] + [0.376, 0.047, 0.188] + [0.058, 0.203, 0.029]
        = [0.458, 0.298, 0.289]
```

### Score Functions

| Name | Formula | Pros | Cons |
|------|---------|------|------|
| **Dot product** | `s · h` | Fast, no parameters | Requires same dimensions |
| **Scaled dot product** | `(s · h) / √d` | Prevents softmax saturation | Used in Transformers |
| **Additive (Bahdanau)** | `vᵀ · tanh(W[s; h])` | Different dimensions OK | More parameters |
| **Multiplicative (Luong)** | `sᵀ W h` | Learnable interaction | More parameters |

**Why scale by √d?**

For high-dimensional vectors, dot products can become very large:
- If d = 512, and each element ~ N(0,1)
- Expected dot product magnitude ~ √512 ≈ 22.6
- Softmax of [22, -22] → [1.0, 0.0] (saturated!)
- Gradients become nearly zero

Scaling by √d keeps values in a reasonable range.

### Types of Attention

**1. Encoder-Decoder Attention** (Cross-Attention)
- Query: decoder state
- Keys/Values: encoder states
- Used in: Translation, summarization

**2. Self-Attention**
- Query, Keys, Values: all from the same sequence
- Each position attends to all positions
- Used in: Transformers (next section)

**3. Causal (Masked) Self-Attention**
- Self-attention but can only attend to past positions
- Used in: GPT, language modeling

---

## Part 6: The Transformer Architecture

### "Attention Is All You Need" (2017)

The transformer (Vaswani et al., 2017) removes recurrence entirely, using only attention. This was revolutionary because:

1. **No sequential dependency**: All positions processed in parallel
2. **Direct connections**: Any position can attend to any other position
3. **Scalable**: Training time is O(1) in sequence length (vs O(n) for RNNs)

```
┌─────────────────────────────────────────────────────────┐
│                    TRANSFORMER                          │
│                                                         │
│  ┌─────────────┐              ┌─────────────┐          │
│  │   ENCODER   │              │   DECODER   │          │
│  │             │              │             │          │
│  │ Self-Attn   │──────────────│ Cross-Attn  │          │
│  │     ↓       │              │     ↓       │          │
│  │ Feed-Fwd    │              │ Self-Attn   │          │
│  │     ↓       │              │     ↓       │          │
│  │   × N       │              │ Feed-Fwd    │          │
│  │             │              │     ↓       │          │
│  └─────────────┘              │   × N       │          │
│        ↑                      └─────────────┘          │
│   Input Embed                       ↑                  │
│   + Position                  Output Embed             │
│                               + Position               │
└─────────────────────────────────────────────────────────┘
```

**Transformer variants**:
- **Encoder-only** (BERT): Bidirectional, good for understanding tasks
- **Decoder-only** (GPT): Autoregressive, good for generation
- **Encoder-Decoder** (T5, BART): Good for seq2seq tasks like translation

### Self-Attention: Every Token Attends to Every Token

Unlike RNN attention (decoder → encoder), self-attention lets each position attend to all positions in the same sequence.

```
Input: "The cat sat"
       [T] [C] [S]
        ↓   ↓   ↓
Q:     q₁  q₂  q₃   (queries: "what am I looking for?")
K:     k₁  k₂  k₃   (keys: "what do I contain?")
V:     v₁  v₂  v₃   (values: "what do I output?")
```

**Attention computation**:

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

1. **QKᵀ**: Compute similarity between all query-key pairs
2. **/ √dₖ**: Scale to prevent softmax saturation
3. **softmax**: Convert to probabilities (per row)
4. **· V**: Weighted sum of values

### Multi-Head Attention

Instead of one attention, use multiple "heads" with different learned projections:

```
head₁ = Attention(QW₁ᵠ, KW₁ᴷ, VW₁ⱽ)
head₂ = Attention(QW₂ᵠ, KW₂ᴷ, VW₂ⱽ)
...
MultiHead = Concat(head₁, head₂, ...) · Wᴼ
```

**Why multiple heads?**
- Different heads can focus on different relationships
- Head 1: syntactic (subject-verb agreement)
- Head 2: semantic (coreference)
- Head 3: positional (nearby words)

### Positional Encoding

Self-attention is permutation-invariant—it doesn't know word order!

**Solution**: Add positional information to embeddings.

**Sinusoidal encoding** (original paper):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Learned positional embeddings** (GPT, BERT):

```
embedding = token_embedding + position_embedding[pos]
```

### Feed-Forward Network

After attention, each position passes through a feed-forward network:

```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
```

Or with GELU (modern):

```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

This is where the model "thinks" about what it learned from attention.

### Layer Normalization and Residual Connections

Each sub-layer has:
1. **Residual connection**: `output = sublayer(x) + x`
2. **Layer normalization**: Normalize across features

```
x → LayerNorm → Attention → + → LayerNorm → FFN → + → output
     └──────────────────────┘    └────────────────┘
           residual                   residual
```

### Causal Masking (Decoder-Only)

For autoregressive generation (GPT), we mask future positions:

```
Attention matrix (before softmax):
        T    h    e    _    c    a    t
    T [ 0   -∞   -∞   -∞   -∞   -∞   -∞ ]
    h [ 0    0   -∞   -∞   -∞   -∞   -∞ ]
    e [ 0    0    0   -∞   -∞   -∞   -∞ ]
    _ [ 0    0    0    0   -∞   -∞   -∞ ]
    c [ 0    0    0    0    0   -∞   -∞ ]
    a [ 0    0    0    0    0    0   -∞ ]
    t [ 0    0    0    0    0    0    0 ]
```

Each position can only attend to itself and previous positions.

---

## Part 7: Why Transformers Won

### Parallelization

| Architecture | Training | Inference |
|--------------|----------|-----------|
| RNN | Sequential (slow) | Sequential |
| Transformer | Parallel (fast) | Sequential* |

*Inference is still autoregressive, but training is fully parallel.

### Scaling Laws

Transformers follow predictable **scaling laws**:

```
Loss ∝ 1 / (Parameters^0.076 × Data^0.095 × Compute^0.050)
```

**Implication**: Double the compute → predictable improvement. No architectural changes needed.

### Long-Range Dependencies

```
RNN: Token 1 → h₁ → h₂ → ... → h₁₀₀ → Token 100
     (100 sequential steps, gradient vanishes)

Transformer: Token 1 ←────attention────→ Token 100
             (1 step, direct connection)
```

### Flexibility

The same architecture works for:
- **Text**: GPT, BERT, T5
- **Images**: ViT (Vision Transformer)
- **Audio**: Whisper
- **Video**: VideoGPT
- **Multimodal**: GPT-4V, Gemini

---

## Summary

| Concept | Key Idea | Limitation Solved |
|---------|----------|-------------------|
| **Word2Vec** | Words as dense vectors from context | Discrete symbols → semantic similarity |
| **CBOW** | Predict center from context | Faster training |
| **Skip-gram** | Predict context from center | Better for rare words |
| **RNN** | Hidden state memory | No sequence modeling |
| **LSTM** | Gated memory cells | Vanishing gradients |
| **Attention** | Dynamic focus on relevant parts | Fixed bottleneck |
| **Transformer** | Self-attention, no recurrence | Sequential processing |

---

## Further Reading

- **Word2Vec**: Mikolov et al., "Efficient Estimation of Word Representations" (2013)
- **LSTM**: Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
- **Attention**: Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
- **Transformer**: Vaswani et al., "Attention Is All You Need" (2017)
- **The Illustrated Transformer**: Jay Alammar's visual guide (jalammar.github.io)

---

## Interview Questions

These questions are commonly asked in ML/NLP interviews. Try to answer them before looking at the hints!

### Conceptual Questions

**Q1: What is the difference between CBOW and Skip-gram in Word2Vec?**

<details>
<summary>Hint</summary>

- CBOW: Predicts center word from context words (faster, better for frequent words)
- Skip-gram: Predicts context words from center word (slower, better for rare words)
- Think about which direction the prediction goes and why that affects rare word handling
</details>

**Q2: Why do RNNs suffer from the vanishing gradient problem? How do LSTMs solve it?**

<details>
<summary>Hint</summary>

- RNNs: Gradients flow through repeated matrix multiplications (Wₕₕ)
- If eigenvalues < 1, gradient shrinks exponentially
- LSTMs: Cell state uses element-wise operations (+, ×), not matrix multiplication
- Forget gate close to 1 allows gradients to flow unchanged
</details>

**Q3: What are the three gates in an LSTM and what does each do?**

<details>
<summary>Hint</summary>

- **Forget gate**: Decides what to remove from cell state (0 = forget, 1 = keep)
- **Input gate**: Decides what new information to add to cell state
- **Output gate**: Decides what to output from cell state to hidden state
</details>

**Q4: Why does the Transformer use scaled dot-product attention (divide by √d)?**

<details>
<summary>Hint</summary>

- For high-dimensional vectors, dot products can become very large
- Large values cause softmax to saturate (output nearly 0 or 1)
- Saturated softmax has near-zero gradients
- Scaling by √d keeps values in a reasonable range
</details>

**Q5: What is the purpose of positional encoding in Transformers?**

<details>
<summary>Hint</summary>

- Self-attention is permutation-invariant (doesn't know word order)
- "cat sat mat" and "mat sat cat" would produce the same attention weights
- Positional encoding adds position information to embeddings
- Can be learned (GPT) or sinusoidal (original Transformer)
</details>

**Q6: What is the difference between encoder-only, decoder-only, and encoder-decoder Transformers?**

<details>
<summary>Hint</summary>

- **Encoder-only** (BERT): Bidirectional attention, good for understanding (classification, NER)
- **Decoder-only** (GPT): Causal attention, good for generation
- **Encoder-Decoder** (T5): Best for seq2seq tasks (translation, summarization)
</details>

### Technical Questions

**Q7: Given a vocabulary of 50,000 words and embedding dimension of 300, how many parameters does the Word2Vec model have?**

<details>
<summary>Hint</summary>

- Two weight matrices: W_in and W_out
- Each is (vocab_size × embedding_dim) = 50,000 × 300
- Total: 2 × 50,000 × 300 = 30,000,000 parameters (30M)
</details>

**Q8: What is the computational complexity of self-attention with respect to sequence length n?**

<details>
<summary>Hint</summary>

- Attention computes QKᵀ which is (n × d) × (d × n) = (n × n)
- This is O(n²) in sequence length
- For n = 4096 tokens, that's 16 million attention scores per layer!
- This is why long-context models are expensive
</details>

**Q9: In multi-head attention with 8 heads and d_model = 512, what is the dimension of each head?**

<details>
<summary>Hint</summary>

- d_head = d_model / n_heads = 512 / 8 = 64
- Each head operates on 64-dimensional projections
- Outputs are concatenated: 8 × 64 = 512 (back to d_model)
</details>

**Q10: Why do we use layer normalization instead of batch normalization in Transformers?**

<details>
<summary>Hint</summary>

- Batch norm: Normalizes across batch dimension (depends on batch statistics)
- Layer norm: Normalizes across feature dimension (independent of batch)
- For variable-length sequences, batch statistics are unstable
- Layer norm works better for sequential data and small batches
</details>

### Coding Questions

**Q11: Implement scaled dot-product attention in NumPy.**

```python
def scaled_dot_product_attention(Q, K, V):
    """
    Q: (seq_len, d_k)
    K: (seq_len, d_k)
    V: (seq_len, d_v)
    Returns: (seq_len, d_v)
    """
    # Your code here
    pass
```

<details>
<summary>Solution</summary>

```python
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # (seq_len, seq_len)
    weights = softmax(scores, axis=-1)  # (seq_len, seq_len)
    output = weights @ V  # (seq_len, d_v)
    return output
```
</details>

**Q12: Implement the LSTM cell state update equations.**

```python
def lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo):
    """
    Returns: h_t, c_t
    """
    # Your code here
    pass
```

<details>
<summary>Solution</summary>

```python
def lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo):
    # Concatenate input and previous hidden state
    concat = np.concatenate([h_prev, x_t])
    
    # Gates
    f_t = sigmoid(Wf @ concat + bf)  # Forget gate
    i_t = sigmoid(Wi @ concat + bi)  # Input gate
    c_tilde = np.tanh(Wc @ concat + bc)  # Candidate
    o_t = sigmoid(Wo @ concat + bo)  # Output gate
    
    # Cell state update
    c_t = f_t * c_prev + i_t * c_tilde
    
    # Hidden state
    h_t = o_t * np.tanh(c_t)
    
    return h_t, c_t
```
</details>

**Q13: Add causal masking to the attention function.**

<details>
<summary>Solution</summary>

```python
def causal_attention(Q, K, V):
    d_k = Q.shape[-1]
    seq_len = Q.shape[0]
    
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Create causal mask (upper triangular = -inf)
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    scores = scores + mask
    
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output
```
</details>

### System Design Questions

**Q14: You're building a sentiment analysis system. Would you use BERT or GPT? Why?**

<details>
<summary>Hint</summary>

- BERT (encoder-only): Better for classification tasks
- Bidirectional attention sees full context
- GPT is autoregressive, designed for generation
- For sentiment: BERT or fine-tuned encoder model
</details>

**Q15: How would you handle a 10,000-word document with a Transformer that has a 4096 token context limit?**

<details>
<summary>Hint</summary>

Options:
1. **Chunking**: Split into overlapping chunks, process separately, aggregate
2. **Hierarchical**: Summarize chunks, then process summaries
3. **Sparse attention**: Use models like Longformer with local + global attention
4. **Retrieval**: Use RAG to retrieve relevant chunks
</details>

**Q16: Your Word2Vec model gives poor embeddings for rare words. How would you improve it?**

<details>
<summary>Hint</summary>

Options:
1. Use Skip-gram instead of CBOW (better for rare words)
2. Use subword embeddings (FastText) - rare words share subwords with common words
3. Increase negative sampling for rare words
4. Use pre-trained embeddings from larger corpus
</details>
