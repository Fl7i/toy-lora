# toy-lora-signal-interpreter
A toy end-to-end LoRA fine-tuning project using on synthetic investment signal data. Built to demonstrate the fine-tuning workflow for research and evaluation.

## Setup

**Requirements:** Python 3.10+, CUDA-capable GPU (4-bit quantization via bitsandbytes)

```bash
git clone https://github.com/<your-username>/toy-lora-signal-interpreter.git
cd toy-lora-signal-interpreter
pip install -r requirements.txt
```

## Usage

**1. Generate synthetic dataset**
```bash
python scripts/generate_dataset.py
```
Writes `data/train.jsonl`, `data/validation.jsonl`, and `data/test.jsonl`.

**2. Fine-tune**
```bash
python scripts/train.py
```
Saves the LoRA adapter to `outputs/qwen3-signal-lora/`.

**3. Run inference**
```bash
python scripts/inference.py
```

## Project structure

```
data/               # Synthetic JSONL datasets
scripts/
  generate_dataset.py
  train.py
  inference.py
outputs/            # Saved LoRA adapter checkpoints
```
