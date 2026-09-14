# Decision Log

Entries cover non-obvious design decisions with rationale, alternatives considered, and trade-offs accepted.
Each entry is defensible in a live engineering review.

---

## Decision 1: Brand Selection — SpotifyCares

**Decision:** Use SpotifyCares as the target brand.

**Rationale:**
SpotifyCares has ~38,000 usable customer-reply pairs with a natural, clean intent taxonomy (payment, login, bug, content, feature, other). It offers:
- Enough volume for reliable TF-IDF retrieval without embedding APIs
- A natural 50/50 auto-handle vs. escalate split (bugs/features can be handled; payment/login need escalation)
- Low noise: English-dominant, fewer fragment tweets than AmazonHelp or Uber

**Alternatives considered:**
- *AmazonHelp*: 4× larger but intent space too diffuse (would need 20+ intents)
- *AppleSupport*: High quality but hardware-heavy → near-100% escalation, making the metric trivial
- *Delta*: Clear intents but too domain-specific and escalation-dominated

**Trade-off accepted:** SpotifyCares is from 2015–2017; Spotify features have changed. Results may not transfer to the current product era.

---

## Decision 2: TF-IDF for All Retrieval (Not Embeddings)

**Decision:** Use `TfidfVectorizer` (bigrams, 5,000 features) for similarity retrieval.

**Rationale:**
- Deterministic and reproducible — no external embedding API calls needed
- Fully explainable: retrieved examples match because of shared keywords
- Fast enough to index 38,000 documents in <1 second locally
- Avoids a dependency on OpenAI Embeddings + vector DB (FAISS/Chroma)

**Alternatives considered:**
- *OpenAI `text-embedding-ada-002` + FAISS*: Better semantic matching but adds ~$0.05/1k tokens cost, non-deterministic across API versions, and requires a running vector store
- *`sentence-transformers` (SBERT)*: Free but requires 400MB+ model download, GPU recommended for speed

**Trade-off accepted:** TF-IDF may fail on paraphrase-heavy queries ("app is broken" vs. "the music stops"). Accepted because the retrieval is used to *ground* the LLM generator, not as the final answer itself.

---

## Decision 3: LLM Structured Output for Classification (Not Fine-Tuned Model)

**Decision:** Use `gpt-4o-mini` with `beta.chat.completions.parse` and a Pydantic schema for intent classification.

**Rationale:**
- Zero annotation budget: no labeled training set is available for fine-tuning
- Structured output (Pydantic enum) eliminates parsing failures — the API guarantees the response matches the schema
- Few-shot prompting with per-intent rubrics achieves reasonable accuracy without any training
- `gpt-4o-mini` is cheap (~$0.15/1M input tokens) — evaluating 200 examples costs <$0.05

**Alternatives considered:**
- *Fine-tuned DistilBERT / SetFit*: Would need 50–100 labeled examples per class; adds reproducibility burden (model weights, training code, GPU)
- *Simple regex heuristic*: Used as baseline labeler for the golden set, but demonstrated poor cross-label accuracy (e.g., "family plan" triggers `payment_issue` but intent is `account_login`)

**Trade-off accepted:** Requires an API key; will not run fully offline. A keyword heuristic fallback is provided for testing without a key.

---

## Decision 4: Keyword Heuristic Fallback When No API Key

**Decision:** When `GEMINI_API_KEY` is absent, the classifier falls back to a deterministic keyword heuristic (not a fixed "app_bug" constant).

**Rationale:**
The original skeleton returned `"app_bug"` for every message without an API key, making the dummy evaluation completely misleading (all predictions were identical). A keyword heuristic:
- Produces varied predictions for varied inputs
- Makes tests meaningful (test can check specific intent predictions)
- Shows the gap between heuristic and LLM performance

**Alternatives considered:**
- *Always escalate*: Safe but masks the classifier's behavior completely
- *Fixed majority class*: Same as always predicting "other" — uninformative

**Trade-off accepted:** The heuristic intentionally matches how the golden set labels were generated, so heuristic classifier metrics on the golden set are inflated. This is documented as a limitation.

---

## Decision 5: Two-Stage Pipeline (Intent → Retrieve → Generate+Escalate)

**Decision:** Separate intent classification from reply generation. The generator receives the classified intent as an additional input.

**Rationale:**
- Explicit checkpoints: we can inspect and override the intent before generation
- Intent drives retrieval filtering (future work: retrieve only from same-intent historical examples)
- Prevents the generator from "discovering" the intent itself, which hides classification failures
- Each stage is independently testable

**Alternatives considered:**
- *Single-shot agent*: One LLM call returns intent + reply + escalation decision simultaneously. Simpler but non-debuggable; failure modes are hidden inside one prompt.
- *ReAct agent loop*: Overkill for tweet-length messages; hard to evaluate; non-deterministic

**Trade-off accepted:** Two API calls per message doubles latency and cost. Acceptable for a prototype where correctness > speed.

---

## Decision 6: Escalation as Part of Reply Generation (Not a Separate Classifier)

**Decision:** The escalation decision is generated by the same LLM call that drafts the reply, not a separate classifier.

**Rationale:**
- Escalation requires understanding the *full context*: intent + retrieval quality + message specificity
- A single call can reason over all three simultaneously
- Avoids cascading errors: a separate escalation model would need its own training data

**Alternatives considered:**
- *Rule-based escalation*: Pure heuristic (e.g., escalate if intent ∈ {account_login, payment_issue}). Used in the golden set labeler but too coarse — many app bugs also need escalation.
- *Separate LLM escalation classifier*: Double the cost; rules overlap with generation prompt anyway

**Trade-off accepted:** Escalation quality is only as good as the LLM's reasoning. Cannot independently tune escalation without retraining the generator.

---

## Decision 7: Heuristic Labeling for Golden Set (With Documented Limitations)

**Decision:** Label the 200-example golden set using a keyword heuristic rather than manual annotation or a separate LLM.

**Rationale:**
- Manual annotation by the developer takes 2–3 hours and introduces personal labeling bias
- Using LLM to label the golden set introduces self-evaluation bias (the judge and labeler are the same model)
- The keyword heuristic is auditable, deterministic, and reproducible
- Labeled examples with uncertain intent are flagged in the `notes` field

**Alternatives considered:**
- *Full manual labeling*: More accurate but time-consuming and not reproducible without a human in the loop
- *GPT-4o-mini labeling*: Risk of self-confirmation: the same model labels the examples and is then evaluated against those labels

**Trade-off accepted:** The heuristic inflates TF-IDF baseline performance because both the labels and the baseline use similar keyword logic. This is explicitly called out in "What Is Misleading About My Headline Number?" and in `report/report.md`.

---

## Decision 8: Conversation-Level Train/Eval Split

**Decision:** The golden set is built by sampling conversations, and those conversation IDs are excluded from the retrieval training set.

**Rationale:**
- Message-level splitting risks leaking nearly-identical messages from the same thread (e.g., follow-up tweet is in eval, original is in retrieval)
- Conversation-level split guarantees no thread appears in both retrieval and evaluation data

**Alternatives considered:**
- *Random message-level split*: Simple but leaks context from the same conversation thread

**Trade-off accepted:** Conversation-level splitting reduces the retrieval corpus by ~200 conversations (~4,800 messages), which is negligible given the 38,000-example corpus.

---

## Decision 9: TF-IDF Baseline Uses Same Keyword Labels as Golden Set

**Decision:** The TF-IDF + Logistic Regression baseline is trained on pseudo-labels generated by the same keyword heuristic used for the golden set.

**Rationale:**
- Having the same labeling logic for training and evaluation is standard (ground truth must match)
- The baseline demonstrates what performance looks like *relative to the heuristic labeler*
- When real human labels are available, the TF-IDF baseline should be re-evaluated

**Trade-off accepted:** This arrangement will show artificially high TF-IDF accuracy (the model learns the exact keywords that produce the labels). This is documented prominently.

---

## Decision 10: Evaluating Only 50 Golden Examples for LLM Judge (Not All 200)

**Decision:** The LLM judge evaluation runs on a 50-example subset of the 200-example golden set.

**Rationale:**
- Each example requires two API calls (one for the pipeline, one for the judge) = 100 tokens * 50 examples = ~5,000 tokens
- Running all 200 examples would take ~8 minutes and cost ~$0.20
- 50 examples is sufficient to estimate statistical significance for the metrics reported
- The remaining 150 are reserved as an untouched held-out pool

**Alternatives considered:**
- *All 200*: Slightly more statistically robust but 4× slower and costlier
- *Only 20*: Too small for reliable macro-F1 estimation

**Trade-off accepted:** Results are noisier than a full 200-example evaluation. Confidence intervals are not formally computed.

---

## Decision 11: No Vector Store for Retrieval (TF-IDF Matrix Loaded In-Memory)

**Decision:** The TF-IDF matrix is built in-memory at startup from `train_set.csv`, not persisted to a vector store.

**Rationale:**
- A 5,000-feature TF-IDF matrix over 4,800 examples takes <50 MB in memory and fits easily on any laptop
- Persisting to disk (pickle) would require version-pinning scikit-learn (deserialization fails across versions)
- In-memory construction takes ~0.5 seconds — acceptable overhead

**Alternatives considered:**
- *Pickle the TF-IDF model*: Faster startup but fragile across sklearn versions
- *ChromaDB / FAISS*: Persistent storage but adds dependencies and setup steps

**Trade-off accepted:** If the dataset grows to 500,000+ examples, in-memory TF-IDF would be impractical. This design only scales to ~100,000 examples before needing replacement.

---

## Decision 12: Fixed Random Seed (42) Throughout

**Decision:** Use `random_state=42` in every sampling and splitting operation.

**Rationale:**
- Reproducibility is a first-class requirement per the assignment spec
- A fixed seed ensures that `scripts/build_golden_set.py` always produces the same 200-example golden set from the same input data
- Prevents "seed shopping" (trying seeds until metrics look best)

**Alternatives considered:**
- *No fixed seed*: Results vary every run — not reproducible
- *Per-script seeds*: Hard to track; someone could accidentally change one seed

**Trade-off accepted:** A single seed cannot capture variance. If a future run uses a different sklearn version that changes sampling behavior, results will still differ.

---

## Decision 13: Reply Generation With Explicit "Do Not Invent" System Prompt

**Decision:** The generator system prompt explicitly prohibits the model from inventing policies, refund amounts, timelines, or guarantees.

**Rationale:**
- The most dangerous failure mode is a hallucinated auto-handled reply that makes a false promise (e.g., "Your refund will be processed in 3-5 days") when in fact no refund has been approved
- An explicit prohibition in the system prompt measurably reduces hallucination rate in structured output mode
- The LLM judge rubric measures "unsupported claims" specifically to catch violations

**Alternatives considered:**
- *Post-hoc hallucination detector*: A second LLM call checks the output for unsupported claims. More thorough but doubles cost and adds latency.
- *Retrieval-only response*: Just return the most similar historical reply verbatim. Zero hallucination risk but zero generalization.

**Trade-off accepted:** System prompt prohibitions are soft constraints — the model can still violate them. A production system would need a post-hoc grounding check, which is flagged as "next steps."

---

## Decision 14: Using `eval_score.model_dump()` Instead of `.dict()` (Pydantic v2 compatibility)

**Decision:** Updated all serialization calls from `.dict()` (Pydantic v1) to `.model_dump()` (Pydantic v2).

**Rationale:**
- OpenAI SDK ≥1.0 ships with Pydantic v2; `.dict()` is deprecated with a warning and will raise an error in future versions
- The original skeleton used `.dict()`, which would break silently if the OpenAI SDK is upgraded

**Trade-off accepted:** Users on Pydantic v1 (unlikely given OpenAI SDK v1+) would need to revert.

---

## Decision 15: `tqdm` Progress Bar Wrapped Around Golden Set Loop

**Decision:** Wrap the evaluation loop in `tqdm` to show per-example progress.

**Rationale:**
- Without progress feedback, a 50-example LLM loop takes ~60–90 seconds with no indication of progress
- `tqdm` adds zero overhead and eliminates "is it hung?" uncertainty
- Good developer UX is part of being a reviewable SDE project

**Trade-off accepted:** None.
