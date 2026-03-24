"""System prompt templates, level definitions, scenarios, styles, and profiles."""

# ---------------------------------------------------------------------------
# Urgency definitions
# ---------------------------------------------------------------------------
URGENCY_DEFINITIONS = {
    "Low": (
        "Minor inconvenience with no immediate impact on essential services or "
        "finances. The customer can wait days or weeks for resolution without "
        "meaningful disruption to their daily life. Examples: cosmetic app "
        "glitch, small billing query, curiosity about a plan feature."
    ),
    "Medium": (
        "Noticeable disruption to service or finances that requires attention "
        "within days. The customer is experiencing partial service degradation, "
        "an unexpected charge, or a process that is stalled. They can still "
        "function but the issue is causing ongoing inconvenience. Examples: "
        "intermittent connectivity, disputed charge on a bill, delayed number "
        "port."
    ),
    "High": (
        "Severe, time-critical impact — complete loss of essential service, "
        "significant financial harm, risk of regulatory or legal consequences, "
        "or a vulnerability/safety concern. The customer needs immediate "
        "resolution and may face escalating harm if the issue persists. "
        "Examples: total service outage lasting days, unauthorised account "
        "changes, data breach exposure, vulnerable customer unable to contact "
        "emergency services."
    ),
}

# ---------------------------------------------------------------------------
# Emotion definitions
# ---------------------------------------------------------------------------
EMOTION_DEFINITIONS = {
    "Low": (
        "Calm and factual — the customer reports the issue in a composed, "
        "matter-of-fact manner. Language is measured and solution-oriented. "
        "There is little overt frustration or blame, though a single "
        "understated expression of disappointment or weariness may appear. "
        "The overall impression is of someone who is inconvenienced but "
        "keeping their feelings largely in check."
    ),
    "Medium": (
        "A frustrated tone that could be read as either controlled irritation "
        "or mild emotional expression depending on context. The customer is "
        "clearly not happy and may use words like 'frustrated', "
        "'disappointed', or 'unacceptable', but they remain coherent. "
        "Depending on their personality and history with the issue, this "
        "could shade slightly calmer or slightly more heated — the key "
        "quality is that their dissatisfaction is visible but not extreme."
    ),
    "High": (
        "Strong dissatisfaction — the customer is clearly upset, and this "
        "comes through in their language and the way they construct their "
        "message. This can manifest as cold, controlled anger (clipped "
        "sentences, pointed observations, formal ultimatums) or as more "
        "explicit distress (expressions of desperation, repeated emphasis, "
        "threats to escalate). The emotional state should be felt through "
        "word choice and sentence rhythm — not primarily through exclamation "
        "marks, capitalisation, or labels like 'livid' or 'furious'."
    ),
}

# ---------------------------------------------------------------------------
# Scenarios (20)
# ---------------------------------------------------------------------------
SCENARIOS = [
    "Difficulty Cancelling Service",
    "Fraud & Scams",
    "Overcharging & Incorrect Billing",
    "Poor Network Coverage",
    "3G Shutdown Impact",
    "Auto-Renewal Without Consent",
    "Billing After Cancellation",
    "High Early Termination Fees",
    "Ineffective AI / Chatbot Support",
    "Unfulfilled Fix Promises",
    "Long Call-Waiting Times",
    "Wrong Sale Due to Agent Mistake",
    "Loyalty Penalty",
    "Mid-Contract Price Increase",
    "Complete Service Outage",
    "Faulty Hardware / Handset Issues",
    "Hidden Fees & Charges",
    "Lack of Progress Updates",
    "Poor Complaint Handling",
    "Slow Broadband Speeds",
]

# ---------------------------------------------------------------------------
# Style → allowed emotion levels (Low=7, Medium=8, High=6 styles)
# ---------------------------------------------------------------------------
STYLE_EMOTION: dict[str, list[str]] = {
    "Formal professional":            ["Low", "Medium", "High"],
    "Casual conversational":          ["Low", "Medium", "High"],
    "Passive-aggressive / sarcastic": ["Low", "Medium", "High"],
    "Verbose and detailed":           ["Low", "Medium", "High"],
    "Terse and minimal":              ["Low", "Medium", "High"],
    "Narrative / storytelling":       ["Low", "Medium", "High"],
    "Legalistic / rights-aware":      ["Low", "Medium", "High"],
    "Polite but firm":                ["Low", "Medium", "High"],
}

# ---------------------------------------------------------------------------
# Scenario → allowed urgency levels (Low=12, Medium=18, High=10 scenarios)
# ---------------------------------------------------------------------------
SCENARIO_URGENCY: dict[str, list[str]] = {
    "Difficulty Cancelling Service":       ["Low", "Medium"],
    "Fraud & Scams":                       ["Medium", "High"],
    "Overcharging & Incorrect Billing":    ["Low", "Medium", "High"],
    "Poor Network Coverage":               ["Low", "Medium"],
    "3G Shutdown Impact":                  ["Medium", "High"],
    "Auto-Renewal Without Consent":        ["Low", "Medium"],
    "Billing After Cancellation":          ["Medium", "High"],
    "High Early Termination Fees":         ["Medium", "High"],
    "Ineffective AI / Chatbot Support":    ["Low", "Medium"],
    "Unfulfilled Fix Promises":            ["Medium", "High"],
    "Long Call-Waiting Times":             ["Low", "Medium"],
    "Wrong Sale Due to Agent Mistake":     ["Medium", "High"],
    "Loyalty Penalty":                     ["Low", "Medium"],
    "Mid-Contract Price Increase":         ["Low", "Medium"],
    "Complete Service Outage":             ["High"],
    "Faulty Hardware / Handset Issues":    ["Low", "Medium", "High"],
    "Hidden Fees & Charges":               ["Low", "Medium"],
    "Lack of Progress Updates":            ["Low", "Medium"],
    "Poor Complaint Handling":             ["Medium", "High"],
    "Slow Broadband Speeds":               ["Low", "Medium"],
}

# ---------------------------------------------------------------------------
# Writing styles (8) — intentionally orthogonal to emotion
# ---------------------------------------------------------------------------
STYLES = [
    "Formal professional",
    "Casual conversational",
    "Passive-aggressive / sarcastic",
    "Verbose and detailed",
    "Terse and minimal",
    "Narrative / storytelling",
    "Legalistic / rights-aware",
    "Polite but firm",
]

# ---------------------------------------------------------------------------
# Customer profiles (8) — persona for the complaint author
# ---------------------------------------------------------------------------
CUSTOMER_PROFILES = [
    "Young professional, tech-savvy, impatient",
    "Elderly customer, not confident with technology",
    "Small business owner relying on service for livelihood",
    "Parent managing a family plan",
    "Student on a tight budget",
    "Long-term loyal customer (10+ years)",
    "Recently switched from another provider",
    "Vulnerable customer with disability or health condition",
]

# ---------------------------------------------------------------------------
# Complaint history depth (4) — how many prior interactions
# ---------------------------------------------------------------------------
COMPLAINT_HISTORY = [
    "First contact — raising the issue for the first time",
    "Second attempt — raised once before with no resolution",
    "Repeat complainer — has contacted 3-5 times over weeks",
    "Escalation — has exhausted normal channels, requesting manager/ombudsman",
]

# ---------------------------------------------------------------------------
# Three rotating system prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = [
    # Template A
    (
        "You are a complaint-writing assistant. Your task is to produce "
        "realistic customer complaints addressed to a UK telecoms provider. "
        "Write only the complaint text — no labels, headings, metadata, or "
        "preamble. Vary the length naturally; some complaints should be short "
        "and others longer. Use realistic but varied customer names, or omit "
        "the name entirely. Never use placeholder names like 'John Doe'. "
        "Include realistic details: dates of prior contacts and specific "
        "amounts where relevant. Mention prior interactions ('I already "
        "called twice last week'). Let emotional intensity come through "
        "naturally via word choice, sentence rhythm, and tone — not "
        "primarily through formatting tricks. Vary sentence length "
        "dramatically — some complaints should be 2-3 sentences, others "
        "2-3 paragraphs."
    ),
    # Template B
    (
        "Imagine you are different real customers contacting a UK "
        "telecommunications company to complain. For each complaint you "
        "generate, output nothing but the raw complaint message — no titles, "
        "no tags, no explanations. The complaints should feel authentic: "
        "varying in length, tone, and detail. If a name is used, make it "
        "sound genuine and diverse — avoid generic placeholders. Ground each "
        "complaint in specific details: dates, prior call history, specific "
        "amounts or plan names. When emotion is high, write like a genuinely "
        "upset customer — the intensity should come from what they say and "
        "how they say it, not from excessive punctuation alone. Some angry "
        "customers write in clipped, controlled fury; others ramble; others "
        "threaten to go to Ofcom. Vary length dramatically — some "
        "complaints should be just a couple of sentences, others several "
        "paragraphs."
    ),
    # Template C
    (
        "Act as a generator of customer complaint messages for a UK telecoms "
        "company. Each output must read like a genuine message a customer "
        "would send. Provide only the complaint body — do not include any "
        "metadata, labels, or framing text. Let the complaints differ "
        "naturally in length and specificity. Use believable names when "
        "appropriate; never use obvious filler names. Make each complaint "
        "feel grounded: include dates, names of staff spoken to, and "
        "relevant history. Express emotion through authentic language — the "
        "reader should sense the customer's emotional state from their word "
        "choices, the rhythm of their sentences, and what they choose to "
        "emphasise, rather than from typographic conventions alone. "
        "Vary length dramatically — some complaints should be very brief, "
        "others much longer and more detailed."
    ),
]
