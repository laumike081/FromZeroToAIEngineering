# From Zero to AI Engineering

[![Deploy Book](https://github.com/laumike081/FromZeroToAIEngineering/actions/workflows/deploy-book.yml/badge.svg)](https://github.com/laumike081/FromZeroToAIEngineering/actions/workflows/deploy-book.yml)

📖 **[Read the book online](https://laumike081.github.io/FromZeroToAIEngineering/)**

A practical learning log and playbook for becoming an AI engineer from scratch. This is an open-source Jupyter Book covering LLMs, AI agents, machine learning, and interview prep.

## What's Inside

| Section | Topics |
|---------|--------|
| **LLMs** | How LLMs work, Transformer architecture, Word2Vec, RNN/LSTM, using OpenAI/Claude/local models |
| **AI Agents** | Tool use, function calling, planning, memory, building reliable workflows |
| **Machine Learning** | Core ML concepts, bias/variance, metrics, cross-validation |
| **LeetCode** | Patterns (two pointers, sliding window, DP), deliberate practice |
| **Foundations** | Big-O, data structures, clean Python |

## Quick Start (Local Development)

```bash
# Clone the repo
git clone https://github.com/laumike081/FromZeroToAIEngineering.git
cd FromZeroToAIEngineering

# Create virtual environment and install dependencies
uv venv
uv sync

# Build the book
uv run jupyter-book build book/

# Open in browser
open book/_build/html/index.html  # macOS
# or: xdg-open book/_build/html/index.html  # Linux
```

## Project Structure

```
FromZeroToAIEngineering/
├── book/
│   ├── _config.yml          # Jupyter Book config
│   ├── _toc.yml              # Table of contents
│   ├── intro.md              # Landing page
│   └── chapters/             # All content
│       ├── 40_llms_overview.md
│       ├── 41_LM_deep_dive.md
│       ├── 42_LM_evolution_code_examples.ipynb
│       ├── 43_how_llms_work.ipynb
│       ├── 44_transformer_deep_dive.md
│       ├── 45_llm_figures.md
│       ├── 46_using_llms_in_practice.ipynb
│       └── ...
├── .github/workflows/        # GitHub Actions for auto-deploy
├── pyproject.toml            # Dependencies
└── README.md
```

## Contributing

This is a personal learning log, but contributions are welcome:

1. Fork the repo
2. Create a branch (`git checkout -b feature/new-chapter`)
3. Make your changes
4. Build locally to verify (`uv run jupyter-book build book/`)
5. Submit a PR

## Auto-Deployment

The book automatically deploys to GitHub Pages on every push to `main`. 

To enable this for your fork:
1. Go to **Settings → Pages**
2. Set Source to **GitHub Actions**
3. Push to `main` and the workflow will deploy

## License

MIT License - feel free to use this as a template for your own learning journey!

## Acknowledgments

- [Jupyter Book](https://jupyterbook.org/) for the publishing framework
- The open-source AI/ML community for countless learning resources
