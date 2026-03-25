import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")
MAX_LENGTH = 192
LABEL_NAMES = ["Low", "Medium", "High"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model (must match train.py architecture exactly) ─────────────────────────
class DeBERTaMultiHead(nn.Module):
    def __init__(self, config_path, num_classes=3):
        super().__init__()
        config            = AutoConfig.from_pretrained(config_path)
        self.backbone     = AutoModel.from_config(config)   # structure only, weights loaded from state dict
        hidden_size       = config.hidden_size
        self.urgency_head = nn.Linear(hidden_size, num_classes)
        self.emotion_head = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return self.urgency_head(cls), self.emotion_head(cls)

# ── Load model & tokenizer ───────────────────────────────────────────────────
print(f"Loading model from '{MODEL_DIR}' ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = DeBERTaMultiHead(MODEL_DIR).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_weights.pt"), map_location=DEVICE))
model.eval()
print(f"Model loaded. Running on {DEVICE}.\n")

# ── Adversarial test cases ───────────────────────────────────────────────────
TESTS = [
    {
        "id": 1,
        "name": "Sarcastic Praise",
        "text": "Absolutely wonderful. Your engineer just pushed an update that wiped my entire call history and reset all my preferences. Truly a masterpiece of customer service. Give whoever approved that release a promotion.",
        "expected_urgency": "Medium",
        "expected_emotion": "High",
        "why": "Glowing words like 'wonderful' and 'masterpiece' mask intense frustration about data loss.",
    },
    {
        "id": 2,
        "name": "Polite Catastrophe",
        "text": "Good morning. I hope this message finds you well. I wanted to let you know that our office broadband has been completely down since 6am and we have 40 staff unable to work or access any systems. No rush, just flagging it when you get a chance.",
        "expected_urgency": "High",
        "expected_emotion": "Low",
        "why": "A full business outage affecting 40 people described with total calm and politeness.",
    },
    {
        "id": 3,
        "name": "Aggressive Triviality",
        "text": "THIS IS ABSOLUTELY DISGUSTING!!! THE FONT ON MY MONTHLY BILL PDF IS SLIGHTLY DIFFERENT FROM LAST MONTH!!! THIS IS NOT WHAT I PAY FOR!!! I WANT A MANAGER NOW!!!",
        "expected_urgency": "Low",
        "expected_emotion": "High",
        "why": "Maximum fury over a cosmetic billing PDF change with zero real impact.",
    },
    {
        "id": 4,
        "name": "Passive-Aggressive Legal Threat",
        "text": "As your SLA guarantees 99.9% uptime and we have now experienced 14 hours of downtime this month, I trust you are already preparing the appropriate service credit. Our solicitors have been made aware of the situation and I look forward to your formal response.",
        "expected_urgency": "High",
        "expected_emotion": "High",
        "why": "Cold, precise legal language — no exclamation marks, but the threat level is severe.",
    },
    {
        "id": 5,
        "name": "Rambling Confusion",
        "text": "Hi there, I was trying to top up my account and I got a bit confused with the new app layout, my sister uses a different network and she said hers looks different too. Anyway I think I may have pressed the wrong thing? The screen went back to the home page. I'll probably try again later. Hope you're having a nice day!",
        "expected_urgency": "Low",
        "expected_emotion": "Low",
        "why": "Long message suggesting a serious issue, but it's a mild self-resolved navigation confusion.",
    },
    {
        "id": 6,
        "name": "Urgent but Cheerful",
        "text": "Hey team! Big fans of the new app redesign! Quick heads up though — it looks like all outgoing calls are failing across the network right now and none of our customers can get through to us! Super keen to get this sorted when you can! Thanks so much!",
        "expected_urgency": "High",
        "expected_emotion": "Low",
        "why": "Complete call failure reported with enthusiasm and exclamation points by a happy customer.",
    },
    {
        "id": 7,
        "name": "Sincere Heartbreak",
        "text": "I've been with you for nine years and always defended you to friends who switched providers. Finding out you've quietly removed the data rollover feature without any notice genuinely hurts. I feel like a loyal customer means nothing to you anymore.",
        "expected_urgency": "Low",
        "expected_emotion": "High",
        "why": "Deep loyalty and sadness over a minor plan feature removal — emotion without urgency.",
    },
    {
        "id": 8,
        "name": "Pure Machine Logic",
        "text": "BGP session dropped on PE-router LON-CORE-04. AS path withdrawal propagating. Estimated 8,000 residential lines affected in SE postcode region. Failover to secondary path unsuccessful. Immediate escalation to NOC required.",
        "expected_urgency": "High",
        "expected_emotion": "Low",
        "why": "Pure network engineering jargon — zero emotional language, but a massive regional outage.",
    },
    {
        "id": 9,
        "name": "Ambiguous Broken",
        "text": "The signal keeps dropping again. It's been doing this on and off for a couple of days now.",
        "expected_urgency": "Medium",
        "expected_emotion": "Medium",
        "why": "True Medium/Medium baseline — mild recurring issue with a hint of frustration but no detail.",
    },
    {
        "id": 10,
        "name": "Overreaction to a Fix",
        "text": "Oh wow, you finally corrected the wrong data allowance displayed on my account page. Only took four months of emails. I hope the team celebrated with champagne. Truly historic engineering work.",
        "expected_urgency": "Low",
        "expected_emotion": "High",
        "why": "Issue already resolved — sarcasm signals high residual anger despite zero current urgency.",
    },
]

# ── Inference helper ─────────────────────────────────────────────────────────
def predict(text):
    enc = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids)).to(DEVICE)

    with torch.no_grad():
        urg_logits, emo_logits = model(input_ids, attention_mask, token_type_ids)

    urg_probs = torch.softmax(urg_logits, dim=-1).squeeze().cpu().tolist()
    emo_probs = torch.softmax(emo_logits, dim=-1).squeeze().cpu().tolist()
    urg_pred  = LABEL_NAMES[int(torch.argmax(urg_logits))]
    emo_pred  = LABEL_NAMES[int(torch.argmax(emo_logits))]
    return urg_pred, emo_pred, urg_probs, emo_probs

# ── Run tests ────────────────────────────────────────────────────────────────
from datetime import datetime

passed = 0
lines = []

def out(text=""):
    print(text)
    lines.append(text)

out("=" * 70)
out("10-ITEM ADVERSARIAL TEST")
out(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
out(f"Model dir : {MODEL_DIR}")
out("=" * 70)

for t in TESTS:
    urg_pred, emo_pred, urg_probs, emo_probs = predict(t["text"])
    urg_ok = urg_pred == t["expected_urgency"]
    emo_ok = emo_pred == t["expected_emotion"]
    both_ok = urg_ok and emo_ok
    if both_ok:
        passed += 1

    status = "PASS" if both_ok else "FAIL"
    out(f"\n[{t['id']:02d}] {t['name']}  [{status}]")
    out(f"     Text: \"{t['text'][:80]}{'...' if len(t['text']) > 80 else ''}\"")
    out(f"     Urgency  — expected: {t['expected_urgency']:6s} | predicted: {urg_pred:6s} {'OK' if urg_ok else 'WRONG'}")
    out(f"              probs  Low={urg_probs[0]:.2f}  Med={urg_probs[1]:.2f}  High={urg_probs[2]:.2f}")
    out(f"     Emotion  — expected: {t['expected_emotion']:6s} | predicted: {emo_pred:6s} {'OK' if emo_ok else 'WRONG'}")
    out(f"              probs  Low={emo_probs[0]:.2f}  Med={emo_probs[1]:.2f}  High={emo_probs[2]:.2f}")
    out(f"     Why tricky: {t['why']}")

out("\n" + "=" * 70)
out(f"RESULT: {passed}/10 passed")
out("=" * 70)

# ── Save output ───────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir  = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, f"adversarial_results_{timestamp}.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nResults saved to '{out_path}'")
