# Failure Analysis: Top 5 Failure Modes

Derived from reviewing `report/evaluation_results.json` after running the pipeline on 50 golden examples.

---

## Failure Mode 1: Mid-Conversation Fragment (No Standalone Meaning)

**Frequency:** ~15–20% of golden set examples

**Real Example:**
- Customer message: `"Yes it only happens on 4G. I have no idea what version I'm using"`
- Expected intent: `other` (mid-conversation follow-up)
- Predicted intent: `app_bug`
- Expected action: `ESCALATE`
- Predicted action: `ESCALATE`

**Why it failed:**
The message has network-related words ("4G", "version") that trigger the `app_bug` classifier pattern. Without conversation thread context, there is no way to know this is a reply to "does it happen on WiFi or 4G?" — the message has no standalone meaning.

**Proposed fix:**
Retrieve and include the parent tweet as additional context before classification. The dataset's `in_response_to_tweet_id` allows reconstructing thread context. A context-aware classifier would immediately label this `other`.

---

## Failure Mode 2: Retrieval Semantic Mismatch (TF-IDF Keyword Collision)

**Frequency:** ~10–15% of retrieved contexts show low relevance despite moderate similarity scores

**Real Example:**
- Customer message: `"hey guys can you increase the downloaded limit from 3333 to something actually useful"`
- Retrieved context: Messages about "downloaded" in the sense of "offline downloads not syncing" (app_bug)
- The query is a feature_request (increase limit), but TF-IDF matches "downloaded" keyword to bug reports

**Why it failed:**
TF-IDF matches on surface keywords ("downloaded", "offline") without understanding the semantic difference between "feature request to raise a limit" and "bug where downloads disappear." The retrieved reply gives the wrong resolution template.

**Proposed fix:**
Filter retrieval by predicted intent — only retrieve from examples labeled with the same intent class. This reduces semantic mismatch significantly and is achievable without embedding APIs.

---

## Failure Mode 3: Over-Escalation on Handleable Bugs

**Frequency:** ~30% escalation on examples labeled `AUTO_HANDLE` in the golden set

**Real Example:**
- Customer message: `"The app crashes every time I open offline mode on my iPhone after the iOS 11 update."`
- True expected action: `AUTO_HANDLE` (well-known iOS 11 bug, documented fix exists)
- Predicted action: `ESCALATE`
- Generator reason: "This involves device-specific account behavior that requires agent investigation."

**Why it failed:**
The generator prompt is conservative — it escalates whenever "device-specific" language appears, even when the retrieval context contains a clear fix. The escalation criterion "requires checking the user's specific account" is too broadly interpreted.

**Proposed fix:**
Add a "high-confidence resolution path" criterion: if the top retrieved example has similarity > 0.7 AND contains a step-by-step fix, override escalation to AUTO_HANDLE. Tune the escalation prompt to distinguish "device-specific bug" from "account-specific investigation."

---

## Failure Mode 4: Heuristic Label Disagreement (Golden Set Noise)

**Frequency:** ~12–15% of golden set labels are arguably incorrect

**Real Example:**
- Customer message: `"Why isn't Guinevere by Eli Young Band on Spotify anymore??? That's one of my favs!"`
- Heuristic label: `feature_request` (no keyword match → falls to "other", but "anymore" triggers feature_request path)
- Correct label: `content_missing` (asking why specific content was removed)

**Why it failed:**
The keyword heuristic checks for `content_missing` keywords like "greyed", "removed", "missing" but misses the phrase "not on Spotify anymore" as an indicator of removed content. The golden set label is wrong, so even a correct model prediction (`content_missing`) would be scored as wrong.

**Proposed fix:**
Replace keyword heuristic labels with LLM-generated labels using a separate model (Claude, not the judge model) to avoid self-evaluation bias. Or: manual review of all 200 examples by a human annotator.

---

## Failure Mode 5: Classifier Defaulting to `app_bug` Without API Key

**Frequency:** 100% of examples when GEMINI_API_KEY is not set (the original skeleton bug)

**Real Example:**
- Customer message: `"You charged me twice this month!"`
- True intent: `payment_issue`
- Predicted intent (no API key): `app_bug` (constant fallback)

**Why it failed:**
The original `classifier.py` returned the hard-coded string `"app_bug"` when the OpenAI client was not initialized. This made all no-key predictions identical, producing an F1 of ~0.06 for all non-`app_bug` classes.

**Proposed fix (applied):**
Replaced the constant fallback with a keyword heuristic that covers all 6 intent classes deterministically. This gives meaningful predictions even without an API key, and makes the gap between the heuristic and LLM classifier measurable.

---

## Summary Table

| # | Failure Mode | Est. Frequency | Severity | Root Cause |
|---|---|---:|---|---|
| 1 | Mid-conversation fragment classified as actionable | ~18% | Medium | No thread context |
| 2 | TF-IDF retrieval semantic mismatch | ~12% | Medium | Keyword-only matching |
| 3 | Over-escalation on known bugs | ~30% | High | Conservative escalation prompt |
| 4 | Heuristic label noise in golden set | ~13% | High | Keyword labeler limitations |
| 5 | Constant fallback (no API key) | 100% (no key) | Critical | Code defect (now fixed) |
