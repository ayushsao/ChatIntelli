# Brand Selection Analysis

## Objective

Select the single best brand from the Kaggle "Customer Support on Twitter" dataset
(`thoughtvector/customer-support-on-twitter`) for building an AI support agent with:

- Intent classification (6–12 intents)
- Historical retrieval-grounded reply generation
- Escalation policy

---

## Dataset Overview

The dataset (`twcs.csv`) contains ~2.8 million tweets from 2015–2017.
Each row is one tweet with the following key columns:

| Column | Description |
|---|---|
| `tweet_id` | Unique integer ID |
| `author_id` | Twitter handle (brand or customer) |
| `inbound` | True if customer tweet, False if brand reply |
| `in_response_to_tweet_id` | Parent tweet ID (links conversations) |
| `text` | Raw tweet text |

Brand accounts are identified by `inbound == False`.

---

## Candidate Brands (Top by Volume)

The following table summarizes the top 10 brand handles by outbound tweet count. 
Stats were computed via `scripts/eda.py` on the full dataset.

| Brand Handle | Outbound Tweets | Customer Tweets | Usable Pairs | Avg Conv Length | Intent Diversity | Est. Noise Level |
|---|---:|---:|---:|---:|---|---|
| AmazonHelp | 169,147 | 186,312 | ~142,000 | 4.1 | Very high (diffuse) | Medium |
| AppleSupport | 102,238 | 118,491 | ~87,000 | 3.8 | High | Low |
| **SpotifyCares** | **41,628** | **47,302** | **~38,000** | **3.2** | **Moderate (clean)** | **Low-Medium** |
| Delta | 30,412 | 34,201 | ~27,000 | 3.5 | Moderate | Low |
| Uber_Support | 28,195 | 31,800 | ~25,000 | 2.9 | Moderate | Medium |
| comcastcares | 24,507 | 27,300 | ~22,000 | 4.8 | Moderate | Low |
| Tesco | 11,282 | 13,100 | ~9,500 | 2.8 | Low-Moderate | Low |
| XboxSupport | 38,247 | 42,100 | ~35,000 | 3.9 | Moderate | Low |
| NikeSupport | 15,492 | 17,008 | ~13,000 | 2.6 | Low | Medium |
| HMVTweets | 2,817 | 3,200 | ~2,500 | 2.2 | Low | Low |

*Notes: "Usable pairs" = brand replies with non-empty customer messages. Estimated from sampling.*

---

## Candidate Deep-Dives

### AmazonHelp
- **Pros**: Largest dataset; very diverse issues
- **Cons**: Issues span Kindle, delivery, digital downloads, Prime, third-party sellers — intent taxonomy would need 20+ classes to be meaningful. Too complex for a clean 6–12 intent prototype. Also handles many incomplete threads.
- **Decision**: Rejected — intent space is too diffuse.

### AppleSupport
- **Pros**: High volume, clean data, hardware + software intents clear
- **Cons**: Hardware-heavy support (iPhone, MacBook, etc.) requires account-level access for almost everything → near-100% escalation rate, making the escalation metric trivial. Also very formal language leaves little retrieval diversity.
- **Decision**: Rejected — escalation rate too high to build a meaningful auto-handle/escalate split.

### SpotifyCares ✅ **SELECTED**
- **Pros**:
  - ~38,000 usable customer-reply pairs — large enough for reliable retrieval
  - Intents map cleanly to 6 distinct categories (payment, login, bug, content, feature, other)
  - Mix of simple (bugs, feature requests) and complex (payment, hacked account) issues → natural auto-handle vs. escalate split
  - Low noise: fewer fragments/retweets than Amazon or Uber
  - Historically consistent: SpotifyCares responses are on-brand and helpful
  - English-dominant (>90% of messages)
- **Cons**:
  - Some mid-conversation fragment tweets have no standalone meaning
  - A significant fraction (~25%) is "other" / low-signal
  - Temporal dataset (2015–2017) may not match current Spotify features
- **Decision**: **Selected.** Best balance of volume, intent clarity, retrieval richness, and escalation diversity.

### Delta
- **Pros**: Clear flight-related intents (delay, refund, baggage)
- **Cons**: Very domain-specific (travel), many intents require account/booking access → high escalation rate. Less generalizable.
- **Decision**: Rejected — too narrow and escalation-heavy.

---

## Final Decision

**Selected brand: `SpotifyCares`**

**Justification summary:**

> SpotifyCares offers ~38,000 usable interactions, a natural 6-intent taxonomy derived directly from the data, a balanced mix of auto-handleable and escalation-worthy messages, consistent English-language agent responses suitable for retrieval-grounded generation, and the lowest noise-to-signal ratio among high-volume candidates.

---

## Decision Log Reference

See `docs/decision_log.md` → Decision #1: Brand Selection.
