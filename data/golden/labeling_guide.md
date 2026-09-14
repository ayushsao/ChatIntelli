# Golden Set Labeling Guide

## Purpose

This document defines the labeling rules used to create `golden_set.jsonl`.
It must be read before adding or reviewing golden set examples to ensure consistency.

---

## Intent Definitions & Decision Rules

### 1. `payment_issue`
Customer is asking about billing, charges, cancellation, refunds, or subscription renewals.

**Include:**
- "You charged me twice"
- "How do I cancel my Premium?"
- "My payment is failing"
- "I want a refund"
- "Why was I billed X amount?"

**Exclude:**
- Cannot log in (→ `account_login`)
- Premium features broken while logged in (→ `app_bug`)
- Asking about Premium features without a billing complaint (→ `feature_request`)

---

### 2. `account_login`
Customer cannot access their Spotify account.

**Include:**
- "My account was hacked"
- "The password reset link doesn't work"
- "I forgot my password"
- "I can't log in with my Facebook account"
- "My email was changed without my permission"

**Exclude:**
- App crashes on launch (→ `app_bug` if the crash is the main issue)
- Billing questions (→ `payment_issue`)

---

### 3. `app_bug`
The Spotify app itself has a technical malfunction.

**Include:**
- App crashes, freezes, or won't open
- Songs randomly pause or skip
- Downloaded songs disappear
- Playback stops without explanation
- App not loading content on specific OS versions

**Exclude:**
- User dislikes a feature that works as designed (→ `feature_request`)
- Content not on the platform (→ `content_missing`)
- Cannot log in to app (→ `account_login`)

---

### 4. `content_missing`
A specific piece of content is unavailable, greyed out, or removed.

**Include:**
- "Why was this album removed?"
- "This song is greyed out and won't play"
- "This podcast episode has disappeared"
- "The title shows wrong metadata"

**Exclude:**
- App crashing (→ `app_bug`)
- Requesting Spotify to ADD new content they never had (→ `feature_request`)

**Gray zone:**
"Why isn't X on Spotify?" — if the content was previously available and was removed, use `content_missing`. If the content has never been on Spotify & user is requesting it, use `feature_request`.

---

### 5. `feature_request`
Customer is asking for a feature that doesn't currently exist, or providing product feedback.

**Include:**
- "Please add lyrics on TV"
- "Can you bring back the old UI?"
- "I wish you had gapless playback"
- "When will Spotify be available in India?"

**Exclude:**
- Feature exists but is broken (→ `app_bug`)
- Content was present but removed (→ `content_missing`)

---

### 6. `other`
Low-signal, mid-conversation fragments, generic praise/frustration, or off-topic.

**Include:**
- Pure mid-conversation continuations ("Yes, still the same issue", "please check DMs")
- Generic frustration with no actionable content ("get ur shit together")
- Generic praise ("Thanks for the help!")
- Off-topic requests ("Oh Spotify, please come to India")

**Exclude:**
- Any message that, even without thread context, has a discernible complaint or request

---

## Escalation Rules

| Intent | Default Expected Action | Override Condition |
|---|---|---|
| `payment_issue` | `ESCALATE` | Never auto-handle (always account-specific) |
| `account_login` | `ESCALATE` | Never auto-handle (always account-specific) |
| `app_bug` | `AUTO_HANDLE` | Escalate if the bug requires account access |
| `content_missing` | `AUTO_HANDLE` | Escalate if content involves licensing disputes |
| `feature_request` | `AUTO_HANDLE` | Escalate only if user is extremely upset |
| `other` | `ESCALATE` | Can't auto-handle without knowing the issue |

---

## Handling Ambiguous Examples

1. **Multi-intent messages:** Choose the PRIMARY intent (the one requiring the most urgent resolution). Note the secondary intent in `notes`.

2. **Fragment messages with no context:** Label `other` unless the fragment is clearly actionable (e.g., "still crashing" → `app_bug` with `notes: implied from context`).

3. **Sarcasm:** Treat sarcastic complaints as genuine reports. "Love when Spotify pauses every 2 minutes!" → `app_bug`.

4. **Ambiguous billing vs. login:** "I pay for Premium but it says I don't have an account" → `account_login` (the root issue is access, not billing).

---

## Example Difficult Cases

| Message | Label | Reason |
|---|---|---|
| "I can't log in and you also charged me twice" | `account_login` | Multi-intent; login is more urgent |
| "Love when Spotify randomly pauses my music!" | `app_bug` | Sarcasm, clear bug report |
| "Why isn't Guinevere by Eli Young Band on Spotify?" | `content_missing` | Specific content missing (not a new request) |
| "please check DMs" | `other` | No standalone meaning |
| "I got charged 129 pesos instead of 9 pesos" | `payment_issue` | Clear billing discrepancy |
| "please approve my Spotify for Artists account" | `account_login` | Account access request |

---

## Golden Set Format

Each record in `golden_set.jsonl` has:

```json
{
  "id": 123456,
  "conversation_id": 123456,
  "customer_message": "cleaned message text",
  "brand_reply_reference": "what SpotifyCares historically replied",
  "intent": "app_bug",
  "expected_action": "AUTO_HANDLE",
  "notes": "Heuristically labeled. Possible content_missing — app vs missing song ambiguous."
}
```

The `notes` field is crucial for ambiguous examples.
