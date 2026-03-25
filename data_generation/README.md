# Synthetic Complaint Generator

This tool generates realistic synthetic customer complaints about UK telecoms services. Each complaint is pre-labelled with:

- **Urgency level** — how urgently the issue needs resolving (Low / Medium / High)
- **Emotion level** — how emotionally the customer is writing (Low = calm, High = distressed)

We use OpenAI's **GPT-5-mini** to write the text, controlling the scenario, writing style, customer profile, and complaint history for every entry. The result is a dataset of 5,000 complaints ready for NLP model training.

## Why synthetic data?

Rather than collecting and manually labelling thousands of real complaints — slow, expensive, and privacy-sensitive — we instruct the AI to write complaints with specific labels built in from the start. Labels are guaranteed correct by design: we tell the model exactly what urgency and emotion level to express when writing each complaint.

---

## How it works — step by step

### Step 1 — Plan the distribution

The generator creates a **3×3 grid** with urgency on one axis and emotion on the other, giving 9 combinations (cells). Every complaint belongs to exactly one cell.

|  | Low Emotion | Medium Emotion | High Emotion |
|---|---|---|---|
| **Low Urgency** | Calm note about a minor issue | Frustrated note about a minor issue | Angry note about a minor issue |
| **Medium Urgency** | Calm report of slow broadband | Frustrated report of slow broadband | Furious report of slow broadband |
| **High Urgency** | Calm report of a complete outage | Frustrated report of a complete outage | Furious report of a complete outage |

The split across urgency levels reflects reality — most complaints are not emergencies:

| Urgency | Share | Count (out of 5,000) |
|---|---|---|
| Low | 35% | 1,750 |
| Medium | 40% | 2,000 |
| High | 25% | 1,250 |

Emotion is distributed evenly within each urgency group (roughly one third per emotion level).

### Step 2 — Assign characteristics

Every complaint is given four characteristics before the AI writes it:

1. **Scenario** — what the complaint is about (1 of 20 UK telecoms topics)
2. **Writing style** — how the text is written (1 of 8 styles)
3. **Customer profile** — who is writing (1 of 8 personas)
4. **Complaint history** — how many times they have contacted before (1 of 4 depths)

No two complaints in the same grid cell share the same combination of all four characteristics.

**Key constraint — scenario–urgency affinity:** not every scenario fits every urgency level. "Complete Service Outage" can never appear as Low urgency. This is enforced via the affinity map below.

### Step 3 — Write the complaints

GPT-5-mini writes the actual text in **batches of 5** at a time, with up to **10 batches running in parallel**. Each batch includes:
- Urgency and emotion level definitions
- A **CRITICAL TONE instruction** specifying exactly how emotional the writing must sound
- The specific scenario, style, profile, and history for each complaint
- Instructions to vary phrasing, length, and detail

If the model returns fewer complaints than requested, the generator retries automatically (up to 2 times per batch).

**Estimated cost:** ~$1.77 for 5,000 complaints at current GPT-5-mini pricing.

### Step 4 — Save the output

All complaints are assembled and saved to `data/telecoms_complaints.csv` with their labels attached.

---

## The 4 Complaint Dimensions

### Urgency levels

| Level | What it means | Typical example |
|---|---|---|
| **Low** | Minor inconvenience with no immediate impact; can wait days or weeks | A small unexplained charge on a bill |
| **Medium** | Noticeable disruption or partial service loss; needs attention within days | Slow broadband affecting daily work |
| **High** | Severe, time-critical issue; complete loss of service or regulatory risk | A business with no internet access at all |

### Emotion levels

| Level | What it means |
|---|---|
| **Low** | Composed and matter-of-fact. Largely keeping feelings in check, though a single understated note of disappointment may appear. |
| **Medium** | Frustrated tone — intentionally ambiguous, could read as controlled irritation or mild emotional expression. Dissatisfaction is visible but not extreme. |
| **High** | Strong dissatisfaction expressed through word choice and sentence structure. Can manifest as cold controlled anger (clipped sentences, formal ultimatums) or explicit distress. The emotional state should be felt, not announced. |

### Writing styles (8 total)

| Style | Description |
|---|---|
| Formal professional | Structured, business-like language |
| Casual conversational | Relaxed, everyday language |
| Passive-aggressive / sarcastic | Ironic or subtly cutting tone |
| Verbose and detailed | Long, thorough, highly detailed account |
| Terse and minimal | Very brief and straight to the point |
| Narrative / storytelling | Tells the story of events in chronological order |
| Legalistic / rights-aware | References consumer rights, Ofcom regulations, or legal options |
| Polite but firm | Courteous in tone but clear and assertive about expectations |

> All styles are available at all emotion levels — style should not be a predictable signal of emotion.

### Customer profiles (8 total)

| Profile |
|---|
| Young professional — tech-savvy and impatient |
| Elderly customer — not confident with technology |
| Small business owner relying on the service |
| Parent managing a family plan |
| Student on a tight budget |
| Long-term loyal customer (10+ years) |
| Recently switched from another provider |
| Vulnerable customer with a disability or health condition |

### Complaint history depths (4 total)

| Depth | Meaning |
|---|---|
| First contact | Raising the issue for the first time |
| Second attempt | Contacted once before with no resolution |
| Repeat complainer | Contacted 3–5 times over several weeks |
| Escalation | Exhausted normal channels; requesting a manager or the ombudsman |

---

## Scenario–Urgency Affinity Map

| Scenario | Low | Medium | High |
|---|:---:|:---:|:---:|
| Difficulty Cancelling Service | ✓ | ✓ | |
| Fraud & Scams | | ✓ | ✓ |
| Overcharging & Incorrect Billing | ✓ | ✓ | ✓ |
| Poor Network Coverage | ✓ | ✓ | |
| 3G Shutdown Impact | | ✓ | ✓ |
| Auto-Renewal Without Consent | ✓ | ✓ | |
| Billing After Cancellation | | ✓ | ✓ |
| High Early Termination Fees | | ✓ | ✓ |
| Ineffective AI / Chatbot Support | ✓ | ✓ | |
| Unfulfilled Fix Promises | | ✓ | ✓ |
| Long Call-Waiting Times | ✓ | ✓ | |
| Wrong Sale Due to Agent Mistake | | ✓ | ✓ |
| Loyalty Penalty | ✓ | ✓ | |
| Mid-Contract Price Increase | ✓ | ✓ | |
| Complete Service Outage | | | ✓ |
| Faulty Hardware / Handset Issues | ✓ | ✓ | ✓ |
| Hidden Fees & Charges | ✓ | ✓ | |
| Lack of Progress Updates | ✓ | ✓ | |
| Poor Complaint Handling | | ✓ | ✓ |
| Slow Broadband Speeds | ✓ | ✓ | |

**Eligible scenarios per urgency level:** Low — 12 | Medium — 18 | High — 10

---

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your API key**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

3. **Run the generator**
   ```bash
   python data_generation/generate_complaints.py
   ```
   Default: 5,000 complaints. Customise with flags:
   ```bash
   python data_generation/generate_complaints.py --total 1000 --seed 42
   ```
   (`--seed` controls assignment randomness for reproducibility; AI writing still varies each run.)

---

## Output Format

`data/telecoms_complaints.csv` — one row per complaint:

| Column | Description |
|---|---|
| `id` | Complaint number (1 to 5,000) |
| `complaint_text` | The full generated complaint message |
| `intended_urgency` | Low, Medium, or High |
| `intended_emotion` | Low, Medium, or High |
| `scenario` | What the complaint is about |
| `style` | How it is written |
| `profile` | Who is writing it |
| `history` | How many times they have contacted before |

---

## File Descriptions

| File | What it does |
|---|---|
| `generate_complaints.py` | Main script — calls the OpenAI API in parallel batches and saves the output CSV |
| `prompts.py` | Defines all labels, scenarios, styles, profiles, history depths, affinity maps, and the 3 system prompts |
| `taxonomy.py` | Plans the full dataset before generation — builds the grid, distributes assignments, and enforces affinity rules |
| `scenario_urgency_affinity.csv` | The affinity map in CSV format for reference |
