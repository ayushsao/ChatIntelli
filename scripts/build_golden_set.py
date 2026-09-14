import pandas as pd
import json
import os
import random

def assign_heuristic_intent(text):
    """
    Keyword heuristic intent labeler.
    Order matters — more specific / urgent intents are checked first.
    Must stay in sync with KeywordHeuristicClassifier in src/classifier.py.
    """
    text = text.lower()

    # 1. Account / Login
    if any(k in text for k in [
        'hacked', 'login', 'password', 'reset', 'email changed',
        'log in', 'access my account', 'locked out', 'forgot my',
        'approve my spotify', 'spotify for artists',
    ]):
        return 'account_login'

    # 2. Payment
    if any(k in text for k in [
        'charge', 'charged', 'payment', 'premium', 'billing',
        'credit card', 'debit card', 'refund', 'cancel', 'deducted',
        'bill', 'invoice', 'subscription', 'renew',
    ]):
        return 'payment_issue'

    # 3. Content missing
    if any(k in text for k in [
        'greyed', 'grayed', 'missing', 'removed', 'remove',
        'podcast', 'album', 'artist', 'song removed', "can't find",
        'no longer', 'anymore', 'wrong title', 'wrong name',
    ]):
        return 'content_missing'

    # 4. App bug — before feature_request (bug signals are more specific)
    if any(k in text for k in [
        'crash', 'crashes', 'crashing', 'skip', 'pause',
        'offline', 'downloaded', 'freez', 'stop', 'bug', 'glitch',
        "won't play", "won't load", "won't open",
        'not working', "doesn't work", 'broken', 'error',
    ]):
        return 'app_bug'

    # 5. Feature request
    if any(k in text for k in [
        'bring back', 'lyrics', 'ui', 'feature', 'shuffle', 'design',
        'wish', 'would be nice', 'should have', 'when will',
        'please add', 'can you add', 'request',
    ]):
        return 'feature_request'

    return 'other'

def assign_escalate_action(intent, text):
    # Determine expected action based on heuristics for our golden set
    if intent in ['account_login', 'payment_issue']:
        # Most of these require sensitive info or account specific investigation
        return 'ESCALATE'
    if intent == 'other':
        return 'ESCALATE'
    return 'AUTO_HANDLE'

def main():
    df = pd.read_csv('data/spotify_interactions.csv')
    
    # Take a random sample 
    golden_pool = df.sample(200, random_state=42).copy()
    
    golden_data = []
    
    for _, row in golden_pool.iterrows():
        text = row['customer_message_cleaned']
        intent = assign_heuristic_intent(text)
        action = assign_escalate_action(intent, text)
        
        record = {
            'id': row['id'],
            'conversation_id': row['conversation_id'],
            'customer_message': text,
            'brand_reply_reference': row['brand_reply'],
            'intent': intent,
            'expected_action': action,
            'notes': "Heuristically labeled for golden set."
        }
        golden_data.append(record)
        
    os.makedirs('data/golden', exist_ok=True)
    with open('data/golden/golden_set.jsonl', 'w') as f:
        for r in golden_data:
            f.write(json.dumps(r) + '\n')
            
    print(f"Saved {len(golden_data)} items to data/golden/golden_set.jsonl")

    # Now save the REST of the data as the training/retrieval set
    golden_ids = set([r['id'] for r in golden_data])
    train_df = df[~df['id'].isin(golden_ids)]
    train_df.to_csv('data/train_set.csv', index=False)
    print(f"Saved {len(train_df)} items to data/train_set.csv")

if __name__ == '__main__':
    main()
