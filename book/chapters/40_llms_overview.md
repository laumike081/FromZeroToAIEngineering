# What is LLM (Large Language Model)?

A **Large Language Model (LLM)** is a neural network trained on massive amounts of text to predict the next token in a sequence. Despite this simple objective, LLMs develop surprisingly general capabilities: they can answer questions, write code, translate languages, summarize documents, and reason through problems.

**Key characteristics**:
- **Scale**: Billions of parameters trained on trillions of tokens (words/subwords).
- **Self-supervised learning**: No manual labeling needed—the training signal comes from predicting the next word.
- **In-context learning**: LLMs can adapt to new tasks from examples provided in the prompt, without weight updates.
- **Emergent abilities**: Capabilities like arithmetic, chain-of-thought reasoning, and code generation appear as models scale up.

**How they work at a high level**:
1. **Tokenization**: Text is split into tokens (subwords or characters).
2. **Embedding**: Each token becomes a high-dimensional vector.
3. **Transformer layers**: Self-attention and feedforward networks process the sequence, allowing each token to "see" all other tokens.
4. **Prediction**: The model outputs a probability distribution over the next token.
5. **Generation**: Tokens are sampled one at a time, fed back as input, and the process repeats.

**Why this matters for AI engineering**:
- LLMs are the foundation for chatbots, coding assistants, search, and autonomous agents.
- Understanding their strengths (pattern matching, fluency, broad knowledge) and weaknesses (hallucination, lack of true reasoning, context limits) is essential for building reliable systems.

---

## The LLM timeline: from statistical models to frontier AI

Understanding where LLMs came from helps you see why certain techniques work and where the field is heading.

### Visual timeline

```
1950s-1980s    1990s-2000s       2013          2017           2018-2019        2020           2022-2023         2024-2025
    │              │               │              │                │              │                │                │
Rule-based    Statistical      Word2Vec     Transformer      BERT/GPT-1      GPT-3          ChatGPT/GPT-4     Multimodal
   NLP           NLP          embeddings   "Attention Is    Pre-train +    Few-shot        RLHF, Agents     Claude 3, Gemini
                                           All You Need"    Fine-tune       prompting       Tool use          o1 reasoning
```

### Era 1: Rule-based and statistical NLP (1950s–2012)

- **Rule-based systems**: Hand-crafted grammars, pattern matching. Powerful in narrow domains, brittle at scale.
- **Statistical NLP**: n-gram language models, Hidden Markov Models, Conditional Random Fields. Required heavy feature engineering.
- **Limitation**: Models couldn't generalize well; each task needed custom pipelines.

### Era 2: Neural word embeddings (2013–2016)

![Word2Vec concept](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Word2vec_training_model.svg/400px-Word2vec_training_model.svg.png)

*Figure: Word2Vec training architecture (Wikimedia Commons)*

- **Word2Vec (2013)**: Words as dense vectors; similar words have similar vectors.
- **GloVe (2014)**: Global word co-occurrence statistics.
- **Key insight**: Distributed representations capture semantic relationships (e.g., "king - man + woman ≈ queen").
- **Limitation**: Fixed embeddings; no context sensitivity ("bank" means the same in "river bank" and "bank account").

### Era 3: Sequence models and attention (2014–2017)

- **RNN/LSTM/GRU**: Process sequences token by token, maintaining hidden state.
- **Seq2Seq + Attention (2014–2015)**: Encoder-decoder models for translation; attention lets the decoder "look back" at encoder states.
- **Limitation**: Sequential processing is slow; long-range dependencies are hard to learn.

### Era 4: The Transformer revolution (2017)

![Transformer architecture](https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png)

*Figure: Original Transformer architecture from "Attention Is All You Need" (Wikimedia Commons)*

**"Attention Is All You Need" (Vaswani et al., 2017)** introduced the Transformer:

- **Self-attention**: Every token can attend to every other token in parallel.
- **Positional encoding**: Inject position information since there's no recurrence.
- **Multi-head attention**: Multiple attention patterns learned simultaneously.
- **Why it mattered**:
  - Parallelizable (fast training on GPUs/TPUs)
  - Better at long-range dependencies
  - Scales predictably with more data and compute

### Era 5: Pre-train + fine-tune paradigm (2018–2019)

| Model | Year | Key Innovation |
|-------|------|----------------|
| **ELMo** | 2018 | Contextualized embeddings from bidirectional LSTMs |
| **GPT-1** | 2018 | Decoder-only Transformer, generative pre-training |
| **BERT** | 2018 | Encoder-only, masked language modeling, bidirectional context |
| **GPT-2** | 2019 | Larger GPT, showed emergent zero-shot capabilities |
| **T5** | 2019 | Text-to-text framework, unified task formulation |

**Key insight**: Pre-train on massive unlabeled text, then fine-tune on specific tasks with much less labeled data.

### Era 6: Scale and emergence (2020–2021)

![GPT-3 scaling](https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Gpt-3-training-examples-702x336.png/640px-Gpt-3-training-examples-examples-702x336.png)

*Figure: GPT-3 few-shot learning examples*

- **GPT-3 (2020)**: 175B parameters. Demonstrated **few-shot prompting**—no fine-tuning needed for many tasks.
- **Scaling laws**: Performance improves predictably with more parameters, data, and compute.
- **Emergent abilities**: Capabilities that appear suddenly at certain scales (e.g., arithmetic, chain-of-thought reasoning).

### Era 7: Alignment and chat (2022–2023)

| Model | Year | Key Innovation |
|-------|------|----------------|
| **InstructGPT** | 2022 | RLHF (Reinforcement Learning from Human Feedback) |
| **ChatGPT** | 2022 | Conversational interface, RLHF-tuned GPT-3.5 |
| **GPT-4** | 2023 | Multimodal (text + images), stronger reasoning |
| **Claude 2** | 2023 | Constitutional AI, longer context (100K tokens) |
| **Llama 2** | 2023 | Open-weights, commercially usable |

**Key techniques**:
- **RLHF**: Train a reward model from human preferences, then optimize the LLM to maximize that reward.
- **Constitutional AI**: Self-critique and revision based on principles.
- **Longer context windows**: From 4K → 8K → 32K → 100K+ tokens.

### Era 8: Frontier models and agents (2024–2025)

| Model | Year | Key Innovation |
|-------|------|----------------|
| **Claude 3** | 2024 | Opus/Sonnet/Haiku tiers, strong reasoning |
| **GPT-4o** | 2024 | Native multimodal (audio, video, text) |
| **Gemini 1.5** | 2024 | 1M+ token context window |
| **o1 / o3** | 2024-2025 | Extended reasoning, "thinking" before answering |
| **Claude 3.5 Sonnet** | 2024 | Computer use, agentic capabilities |
| **DeepSeek-R1** | 2025 | Open-weights reasoning model |

**Current frontiers**:
- **Extended reasoning**: Models that "think step by step" internally before responding.
- **Tool use and agents**: Models that can call APIs, browse the web, write and execute code.
- **Multimodal native**: Single models handling text, images, audio, video.
- **Open-weights competition**: Llama 3, Mistral, Qwen, DeepSeek challenging closed models.

---

## How to use this part

- Keep a small collection of prompts and counterexamples: what works and what fails.
- Write down failure modes: hallucination patterns, formatting issues, refusal issues.
- Treat evaluation as a first-class topic (even if it's a simple checklist at first).

---

## Chapters in this part

- `41_LM_deep_dive`: In-depth conceptual guide to Word2Vec, CBOW, Skip-gram, RNN, LSTM, attention, and transformer architecture.
- `42_LM_evolution_code_examples`: LLM evolution through code - N-gram, Word2Vec, RNN, and Transformer examples.
- `43_how_llms_work`: Step-by-step guide to how LLMs work, with code examples (tokenization, attention, prompting, RAG).
- `44_transformer_deep_dive`: Detailed walkthrough of the Transformer architecture with step-by-step examples and interview questions.
- `45_llm_figures`: Reference diagrams for transformer architecture and attention.
- `46_using_llms_in_practice`: Hands-on guide to using private APIs (OpenAI, Claude, Gemini, Mistral) and open-source models (LLaMA, Qwen).
