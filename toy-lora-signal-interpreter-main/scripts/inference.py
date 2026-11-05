from pathlib import Path
import json
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-0.6B"
ADAPTER = Path(__file__).resolve().parent.parent / "outputs" / "qwen3-signal-lora"

# ==========================================================
# Tokenizer & Exact Quantization Config Load
# ==========================================================
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

# ==========================================================
# Input Setup
# ==========================================================
SYSTEM_PROMPT = (
    "You are a disciplined quantitative research assistant. Generate a structured report "
    "following the exact sections: 1. Signal summary, 2. What changed, 3. Evidence and context, "
    "4. Research interpretation, 5. Limits and uncertainty, 6. Follow-up questions. "
    "CRITICAL: Do not invent market explanations or causal narratives. If no external context "
    "or macro news is supplied, explicitly state that in the Limits and uncertainty section."
)

signal_input = {
    "record_id": "NASDAQ_100_2026-07-09",
    "instrument": "NASDAQ 100",
    "signal": "Long",
    "confidence": 0.87,
    "previous_confidence": 0.81,
    "volatility": "High",
    "breadth": "Positive",
    "correlation_regime": "Moderate",
    "historical_analogue": "Technology-led rally",
    "model_version": "v2.1.4-beta"
}

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": "Interpret the following approved investment signal.\n\n" + json.dumps(signal_input, indent=2),
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

# ==========================================================
# Execution Loop
# ==========================================================
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.1,  # Keep it ultra-low to reinforce deterministic reporting
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)