from enum import Enum
from typing import List, Dict

class Intent(str, Enum):
    PAYMENT_ISSUE = "payment_issue"
    ACCOUNT_LOGIN = "account_login"
    APP_BUG = "app_bug"
    CONTENT_MISSING = "content_missing"
    FEATURE_REQUEST = "feature_request"
    OTHER = "other"

INTENT_DEFINITIONS = {
    Intent.PAYMENT_ISSUE: {
        "name": "Payment & Subscription Issue",
        "description": "User is having issues with billing, premium subscription, payment methods, or refunds.",
        "inclusion_criteria": "Mentions of charges, credit cards, premium not working, canceling subscriptions.",
        "exclusion_criteria": "Cannot log in to the account (that is account_login).",
        "examples": ["You charged me twice this month!", "How do I cancel my premium?"]
    },
    Intent.ACCOUNT_LOGIN: {
        "name": "Account & Login Problem",
        "description": "User cannot access their account, forgot password, or account was hacked.",
        "inclusion_criteria": "Mentions of hacked, locked out, forgot password, email changed.",
        "exclusion_criteria": "Issues with premium features not working while logged in.",
        "examples": ["My account was hacked and email changed.", "I forgot my password and the reset link isn't working."]
    },
    Intent.APP_BUG: {
        "name": "App Bug or Crash",
        "description": "The app is crashing, songs are skipping, offline sync not working, or UI is broken.",
        "inclusion_criteria": "App freezing, crashing, reinstalling doesn't help, offline downloaded songs disappear.",
        "exclusion_criteria": "Complaining about a feature working as intended but user dislikes it.",
        "examples": ["The app crashes every time I open offline mode.", "My songs pause randomly on iOS."]
    },
    Intent.CONTENT_MISSING: {
        "name": "Content Missing",
        "description": "A specific song, album, artist, or podcast is missing, greyed out, or removed.",
        "inclusion_criteria": "Greyed out songs, artist removed, podcast disappeared.",
        "exclusion_criteria": "App crashing (which implies it's a bug).",
        "examples": ["Why did you remove my favorite album?", "This podcast episode is greyed out."]
    },
    Intent.FEATURE_REQUEST: {
        "name": "Feature Request & Feedback",
        "description": "User is asking for a new feature or providing feedback on app design.",
        "inclusion_criteria": "Asking for lyrics on a specific device, requesting UI changes, wanting better shuffle.",
        "exclusion_criteria": "Complaining about a broken feature (that's app_bug).",
        "examples": ["Please bring back the old UI.", "When will we get lyrics on TV?"]
    },
    Intent.OTHER: {
        "name": "Other / Generic",
        "description": "Unclear, junk, spam, generic praise or frustration.",
        "inclusion_criteria": "Anything that doesn't fit the above.",
        "exclusion_criteria": "Clear intent matching above.",
        "examples": ["Thanks for the help!", "You suck."]
    }
}

VALID_INTENTS = [i.value for i in Intent]
