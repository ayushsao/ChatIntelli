import pandas as pd
import kagglehub
import os
import json

path = kagglehub.dataset_download('thoughtvector/customer-support-on-twitter')
csv_path = os.path.join(path, 'twcs', 'twcs.csv')
df = pd.read_csv(csv_path)

# Find all tweets involving SpotifyCares
brand = "SpotifyCares"

# Find all tweets from the brand
brand_tweets = df[df['author_id'] == brand]

# It's better to find conversation threads.
# We can find inbound tweets that are starting a thread targeting the brand, or where the brand replied.
# Let's get customer tweets that SpotifyCares replied to.
customer_tweets = df[df['tweet_id'].isin(brand_tweets['in_response_to_tweet_id'].dropna())]

print(f"Found {len(customer_tweets)} customer tweets that {brand} replied to.")

# Let's sample a few customer -> brand interactions
sample = pd.merge(customer_tweets, brand_tweets, left_on='tweet_id', right_on='in_response_to_tweet_id', suffixes=('_customer', '_brand'))
print(f"Merged interactions: {len(sample)}")

# Print 10 random samples to understand the intents
for i, row in sample.sample(20, random_state=42).iterrows():
    print(f"C: {row['text_customer']}")
    print(f"B: {row['text_brand']}")
    print("-" * 40)
