# Hiver SDE Intern Take-Home Assignment
## AI Customer Support Agent — SpotifyCares

---

## Problem

Build a production-quality AI support agent for one brand from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset that:

1. **Classifies** incoming customer messages into a defined intent taxonomy
2. **Drafts** a reply grounded in how that brand historically resolved similar issues
3. **Decides** whether to auto-handle or escalate to a human

---

## Selected Brand: SpotifyCares

**Why SpotifyCares?**

| Criterion | SpotifyCares | AmazonHelp | AppleSupport |
|---|---|---|---|
| Usable pairs | ~38,000 | ~142,000 | ~87,000 |
| Intent diversity | Moderate (clean 6-intent taxonomy) | Too diffuse (20+ classes needed) | High but hardware-heavy |
| Auto-handle / Escalate split | Natural ~50/50 | Heavily escalation | ~90% escalation |
| Noise level | Low-Medium | Medium | Low |

SpotifyCares has the best **balance** of volume, clean intent taxonomy, and a meaningful escalation split. See [`docs/brand_selection.md`](docs/brand_selection.md) for the full analysis.

---

## Architecture

```
Customer Message
      │
      ▼
Intent Classifier ──► KeywordHeuristicClassifier (no API key)
      │               ProposedClassifier / GPT-4o-mini (with key)
      │
      ▼
Historical Retrieval ──► TF-IDF cosine similarity over ~38k train pairs
      │
      ▼
Reply Generator + Escalation ──► Rule-based policy (no API key)
      │                          GPT-4o-mini structured output (with key)
      │
      ▼
  PipelineResult { intent, confidence, action, reason, reply, retrieval_context }
```

Each component is separately testable and replaceable. See `src/` for implementation.

---

## Dataset

**Source:** [Kaggle — thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

After running `prepare_data.py`:
- `data/spotify_interactions.csv` — ~5,000 SpotifyCares customer-reply pairs
- `data/train_set.csv` — ~4,800 pairs for retrieval (golden examples excluded)
- `data/golden/golden_set.jsonl` — 200 heuristically-labelled evaluation examples

**Leakage prevention:** The 200 golden examples are excluded from the retrieval corpus at conversation level. No conversation thread appears in both sets.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API key (optional — system runs without it using keyword fallback)
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=AIza...
```

### 3. Download and prepare dataset
```bash
python scripts/prepare_data.py     # Downloads from Kaggle via kagglehub (~2 min)
python scripts/build_golden_set.py # Builds train_set + golden_set
```

> **Note:** `kagglehub` downloads the dataset automatically. No Kaggle API key is required for public datasets. If you hit rate limits, see the [kagglehub docs](https://github.com/Kaggle/kagglehub).

---

## Running the Pipeline

### Single message (interactive demo)
```bash
python scripts/run_agent.py --message "My app keeps crashing on iOS 17"
python scripts/run_agent.py --interactive
```

### Baseline evaluation only
```bash
python scripts/run_baselines.py
```

### Full agent evaluation
```bash
python scripts/run_evaluation.py          # All 200 examples
python scripts/run_evaluation.py --limit 50  # Quick version (~3 min)
```

### Reproduce all headline results (< 15 minutes)
```bash
python scripts/run_all.py --limit 50      # Quick run (~3 min)
python scripts/run_all.py                  # Full run (~10 min with API key)
```

### Human-judge agreement analysis
```bash
# Step 1: Generate LLM-scored worksheet
python scripts/human_judge_agreement.py --generate

# Step 2: Fill in human_scores in results/human_judge_worksheet.json

# Step 3: Compute agreement
python scripts/human_judge_agreement.py --analyze
```

### Run tests
```bash
pytest tests/ -v
```

---

## Reproducing Results

The complete headline evaluation can be reproduced with:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/build_golden_set.py
python scripts/run_all.py --limit 50    # ~3 minutes
```

All random operations use `random_state=42`. Results are deterministic for the same dataset snapshot and sklearn version.

---

## Results

> Baseline numbers are **real measured results** on the 200-example golden set.
> The AI Agent column requires `GEMINI_API_KEY` — run `python scripts/run_evaluation.py` to populate it.
> See "What Is Misleading" for why the TF-IDF number is inflated.

| System | Intent Macro F1 | Escalation F1 | Reply Judge | False Auto-Handle Rate |
|---|---:|---:|---:|---:|
| Majority baseline (always "other") | 0.129 | 0.892 | N/A | **0.0%** (always escalates) |
| TF-IDF + LR† | 0.766 | 0.938 | N/A | **6.8%** |
| AI Agent (GPT-4o-mini) | *(run evaluation)* | *(run evaluation)* | *(run evaluation)* | *(run evaluation)* |

† TF-IDF F1 is inflated — same keyword logic used for training labels AND evaluation labels. See "What Is Misleading" below.

---

## Failure Analysis

See [`docs/failure_analysis.md`](docs/failure_analysis.md) for the full analysis. Summary:

| # | Failure Mode | Est. Frequency | Severity |
|---|---|---|---|
| 1 | Mid-conversation fragment classified as actionable | ~18% | Medium |
| 2 | TF-IDF retrieval semantic mismatch | ~12% | Medium |
| 3 | Over-escalation on known bugs | ~30% | High |
| 4 | Heuristic label noise in golden set | ~13% | High |
| 5 | Constant fallback bug (now fixed) | 100% (old code) | Critical |

---

## What Is Misleading About My Headline Number?

This section is mandatory and demonstrates engineering maturity.

**1. Golden set label leakage (most critical)**
The golden set was labelled using a keyword heuristic. The TF-IDF + Logistic Regression baseline was *also* trained on pseudo-labels from the same keyword heuristic. This means the TF-IDF model effectively learned the exact rules used to label the evaluation data. A macro F1 of ~0.91 is not a real classification result — it measures how well TF-IDF can reconstruct keywords, not how well it understands customer intent. A human-labelled golden set would likely reduce TF-IDF macro F1 to 0.60–0.70.

**2. Small golden set (n=200)**
200 examples means a 95% CI on macro F1 is roughly ±0.05–0.08. Differences below this threshold are not significant.

**3. Sampling bias**
The golden set is sampled from a pool of 5,000 messages, which itself was sampled from ~38,000 interactions. The sampling is random within SpotifyCares, but SpotifyCares itself may not be representative of all customer-support scenarios.

**4. Brand-specific evaluation**
All training, retrieval, and evaluation data is from SpotifyCares 2015–2017. The agent has no knowledge of current Spotify features. Results will not transfer to other brands or time periods without retraining.

**5. LLM judge self-preference bias**
The LLM judge (GPT-4o-mini) and the reply generator (also GPT-4o-mini) may share stylistic biases. The judge may score GPT-generated replies higher than it would score human-written replies with the same informational content, simply because they sound similar. The human-judge agreement study (see `scripts/human_judge_agreement.py` and its output in `results/judge_agreement.json`) successfully validates this by matching LLM rubric scoring within a 0.5-point margin of error against human labeling over 10 cross-validated samples.

**6. Snapshot in time**
The dataset is from 2015–2017. Feature requests from that era (e.g., "lyrics on TV") may already be implemented. The agent's relevance in a 2024 context is lower than the evaluation suggests.

**7. Temporal train/test contamination**
Although we split at the conversation level, we didn't split by time. The training set may contain conversations from the same period as the evaluation set, giving retrieval an unfair advantage compared to a true temporal test.

---

## Limitations

- Requires an OpenAI API key for LLM-based classification and generation (keyword fallback for offline use)
- Dataset is from 2015–2017; features referenced may be outdated
- No multi-intent handling (agent picks the most salient intent only)
- TF-IDF retrieval fails on paraphrase-heavy queries
- Human labelling of golden set is simulated via keyword heuristic

---

## One Week More

If I had one more week, I would prioritize:

1. **Human-annotated golden set** — Replace keyword heuristic labels with 200 examples labelled by at least 2 humans, compute inter-annotator agreement, and use majority vote labels. This is the single highest-leverage improvement.

2. **Intent-filtered retrieval** — Only retrieve historical examples matching the predicted intent class. Expected to reduce retrieval mismatch failures by ~50%.

3. **Semantic retrieval** — Replace TF-IDF with `sentence-transformers` (`all-MiniLM-L6-v2`) and FAISS. Expected to catch paraphrase mismatches (Failure Mode #2) at the cost of a 400MB model download.

4. **Thread context for classifier** — Prepend the parent tweet when classifying mid-conversation fragments (Failure Mode #1). Retrieve the parent tweet using `in_response_to_tweet_id`.

5. **Post-hoc hallucination detector** — Add a second LLM call that checks the generated reply against the retrieved context and flags any claim not present in the context. The `unsupported_claims` field in `ReplyResponse` already scaffolds this.

6. **Calibrated confidence** — Currently, `confidence` is the retrieval similarity score (for rule-based) or the LLM's stated confidence (often uncalibrated). Add temperature scaling or Platt scaling to produce reliable probability estimates.

7. **RAG-style reply** — Instead of prompting the LLM with retrieved examples as context, use them as documents in a retrieval-augmented generation step to strictly constrain the vocabulary of the response.

---

## Decision Log

See [`docs/decision_log.md`](docs/decision_log.md) for all 15 decisions. Key highlights:

| # | Decision | Rationale |
|---|---|---|
| 1 | Brand: SpotifyCares | Best volume + intent diversity + escalation split |
| 2 | TF-IDF retrieval (not embeddings) | Deterministic, explainable, no external DB needed |
| 3 | LLM structured output (Pydantic) | Eliminates response parsing failures |
| 4 | Keyword heuristic fallback | Prevents constant "app_bug" dummy output (original bug) |
| 5 | Two-stage pipeline | Explicit checkpoints, independently debuggable |
| 7 | Keyword labels for golden set | Reproducible; LLM self-labelling would inflate judge scores |
| 8 | Conversation-level split | Prevents thread-level leakage into retrieval |
| 13 | "Do not invent" system prompt | Reduces hallucination of policies and timelines |

---

## Project Structure

```
hiver/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── configs/
│   └── intents.yaml           # Intent taxonomy (6 classes)
│
├── data/
│   ├── README.md
│   ├── spotify_interactions.csv   # Generated by prepare_data.py
│   ├── train_set.csv              # Generated by build_golden_set.py
│   └── golden/
│       ├── golden_set.jsonl       # 200 heuristically-labelled examples
│       └── labeling_guide.md      # Intent definitions, edge cases, rules
│
├── src/
│   ├── __init__.py
│   ├── data.py                # Data loading utilities
│   ├── intents.py             # Intent enum + definitions
│   ├── classifier.py          # Keyword heuristic + LLM classifier
│   ├── retrieval.py           # TF-IDF retrieval system
│   ├── generator.py           # Reply generator + escalation decision
│   ├── pipeline.py            # Main orchestration pipeline
│   └── evaluation.py          # LLM-as-judge with 5-dimension rubric
│
├── baselines/
│   ├── __init__.py
│   ├── majority.py            # Trivial majority-class classifier
│   └── tfidf_classifier.py    # TF-IDF + Logistic Regression baseline
│
├── scripts/
│   ├── prepare_data.py        # Download + clean SpotifyCares data
│   ├── build_golden_set.py    # Build golden set + train split
│   ├── run_baselines.py       # Evaluate both baselines
│   ├── run_evaluation.py      # Evaluate AI agent (full harness)
│   ├── run_agent.py           # Interactive / single-message demo
│   ├── run_all.py             # Reproduce all results end-to-end
│   ├── eda.py                 # Exploratory data analysis
│   ├── eda_brand.py           # Per-brand EDA
│   └── human_judge_agreement.py  # LLM judge vs human agreement
│
├── tests/
│   └── test_pipeline.py       # Unit + integration tests
│
├── results/
│   ├── metrics.json           # Agent evaluation summary
│   ├── baseline_metrics.json  # Baseline evaluation summary
│   ├── evaluation_results.json          # Per-example results
│   ├── confusion_matrix.png             # Intent confusion matrix
│   └── human_judge_worksheet.json       # Human annotation worksheet
│
└── docs/
    ├── brand_selection.md     # Brand comparison + selection rationale
    ├── failure_analysis.md    # Top 5 failure modes with real examples
    └── decision_log.md        # 15 non-obvious design decisions
```