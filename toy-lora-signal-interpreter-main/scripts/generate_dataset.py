import json
import random
from pathlib import Path
from datetime import datetime, timedelta

# ==========================================================
# Configuration
# ==========================================================

SEED = 42
NUM_EXAMPLES = 1000

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Domain Values & Context Maps
# ==========================================================

INSTRUMENTS = ["S&P 500", "NASDAQ 100", "Russell 2000", "Euro Stoxx 50", "Nikkei 225"]
SIGNALS = ["Long", "Short", "Neutral"]

# Shared system instructions used for both training and inference
SYSTEM_PROMPT = (
    "You are a disciplined quantitative research assistant. Generate a structured report "
    "following the exact sections: 1. Signal summary, 2. What changed, 3. Evidence and context, "
    "4. Research interpretation, 5. Limits and uncertainty, 6. Follow-up questions. "
    "CRITICAL: Do not invent market explanations or causal narratives. If no external context "
    "or macro news is supplied, explicitly state that in the Limits and uncertainty section."
)

FOLLOW_UPS = [
    "Review whether the approved breadth metrics changed materially around the signal window.",
    "Verify cross-market correlation shifts against alternative historical point-in-time analogues.",
    "Validate underlying version diagnostic changes relative to the baseline regime.",
]

# ==========================================================
# Semi-Structural Data Generator (Removes Random Noise)
# ==========================================================

def generate_coherent_record(date_str):
    instrument = random.choice(INSTRUMENTS)
    
    # Structural Mapping: tie market diagnostics loosely together
    volatility = random.choice(["Low", "Moderate", "High", "Extreme"])
    
    if volatility in ["High", "Extreme"]:
        breadth = random.choice(["Negative", "Neutral"])
        correlation = random.choice(["Moderate", "High"])
        signal = random.choice(["Short", "Neutral"])
        # Stressed environments usually push baseline model confidence down
        confidence = round(random.uniform(0.35, 0.65), 2)
        previous_confidence = round(random.uniform(0.55, 0.85), 2)
        analogue = random.choice(["Risk-off episode", "Volatility compression", "Credit stress"])
    else:
        breadth = random.choice(["Positive", "Neutral"])
        correlation = random.choice(["Low", "Moderate"])
        signal = random.choice(["Long", "Neutral"])
        confidence = round(random.uniform(0.60, 0.95), 2)
        previous_confidence = round(random.uniform(0.50, 0.80), 2)
        analogue = random.choice(["Momentum rally", "Bullish breakout", "Technology-led rally"])

    return {
        "record_id": f"{instrument.replace(' ', '_')}_{date_str}",
        "instrument": instrument,
        "signal": signal,
        "confidence": confidence,
        "previous_confidence": previous_confidence,
        "volatility": volatility,
        "breadth": breadth,
        "correlation_regime": correlation,
        "historical_analogue": analogue,
        "model_version": "v2.1.4-beta"
    }

# ==========================================================
# Strict Schema Report Builder
# ==========================================================

def build_strict_report(record):
    delta = round(record["confidence"] - record["previous_confidence"], 2)
    change_str = f"Prior confidence {record['previous_confidence']}; change {'' if delta < 0 else '+'}{delta}"
    
    report = [
        "1. Signal summary",
        f"- Record ID: {record['record_id']}",
        f"- Instrument/Horizon: {record['instrument']} / One-day horizon",
        f"- Signal: {record['signal']}",
        f"- Confidence: {record['confidence']}",
        f"- Model Version: {record['model_version']}\n",
        
        "2. What changed",
        f"Deterministically calculated change: {change_str}.\n",
        
        "3. Evidence and context",
        f"- Volatility Regime: {record['volatility']} [Diagnostic_Set]",
        f"- Market Breadth: {record['breadth']} [Diagnostic_Set]",
        f"- Correlation Regime: {record['correlation_regime']} [Diagnostic_Set]",
        f"- Historical Analogue: {record['historical_analogue']} [Reference_Set]\n",
        
        "4. Research interpretation",
        f"The model indicators reflect a {record['signal']} status accompanied by {record['volatility'].lower()} volatility. "
        f"Based solely on the provided inputs, this indicates a tracking profile matching the {record['historical_analogue'].lower()} reference matrix.\n",
        
        "5. Limits and uncertainty",
        "No external news or macro context was supplied. Interpretation is therefore limited strictly to statistical diagnostics. "
        "The available record does not establish an external causal market explanation.\n",
        
        "6. Follow-up questions",
        f"- {random.choice(FOLLOW_UPS)}"
    ]
    return "\n".join(report)

# ==========================================================
# Sequential Generation & Splitting
# ==========================================================

start_date = datetime(2026, 1, 1)
raw_records = []

for i in range(NUM_EXAMPLES):
    current_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    raw_records.append(generate_coherent_record(current_date))

examples = []
for rec in raw_records:
    user_prompt = "Interpret the following approved investment signal.\n\n" + json.dumps(rec, indent=2)
    assistant_response = build_strict_report(rec)
    
    examples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    })

# Point-in-time sequential split (No Shuffling across boundaries)
n_train = int(NUM_EXAMPLES * TRAIN_RATIO)
n_valid = int(NUM_EXAMPLES * VALID_RATIO)

train = examples[:n_train]
validation = examples[n_train:n_train + n_valid]
test = examples[n_train + n_valid:]

def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

save_jsonl(OUTPUT_DIR / "train.jsonl", train)
save_jsonl(OUTPUT_DIR / "validation.jsonl", validation)
save_jsonl(OUTPUT_DIR / "test.jsonl", test)

print(f"Training: {len(train)} | Validation: {len(validation)} | Test: {len(test)}")