# Offline ChatBot LLM

A complete, professional-quality offline chatbot implementation that runs on a single consumer GPU (RTX 3090) with only PyTorch and NumPy as external dependencies.

## Features

- **Decoder-only Transformer** (~50M parameters) implemented from scratch
- **Byte-level BPE tokenizer** trained from scratch
- **Supervised Fine-Tuning (SFT)** with AdamW and cosine scheduling
- **DPO-based RLHF** for preference alignment
- **Safety filtering** with configurable rules
- **CLI chat interface** with multi-turn conversations
- **HTTP API server** using Python stdlib
- **Mixed precision training** support (optional)
- **CPU/GPU compatibility** with automatic device detection

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (only PyTorch and NumPy required)
pip install torch numpy

# Verify installation
python -c "import torch; import numpy; print('PyTorch:', torch.__version__, 'NumPy:', numpy.__version__)"