import pandas as pd
import kagglehub
import os
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    return text.strip()

def main():
    print("Downloading/Locating dataset...")
    path = kagglehub.dataset_download('thoughtvector/customer-support-on-twitter')
    csv_path = os.path.join(path, 'twcs', 'twcs.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(path, 'twcs.csv')
        
    print("Loading datasets (this may take a minute)...")
    df = pd.read_csv(csv_path)

    brand = "SpotifyCares"
    print(f"Extracting {brand} conversations...")
    
    brand_tweets = df[df['author_id'] == brand]
    customer_tweets = df[df['tweet_id'].isin(brand_tweets['in_response_to_tweet_id'].dropna())]

    merged = pd.merge(customer_tweets, brand_tweets, left_on='tweet_id', right_on='in_response_to_tweet_id', suffixes=('_customer', '_brand'))
    
    # We only care about English mostly. Twitter has a lot of languages, but SpotifyCares uses different handles or languages? We can filter heuristically or just keep all.
    # We will sample 5000 examples
    merged = merged.sample(5000, random_state=42)
    
    output_df = pd.DataFrame({
        'id': merged['tweet_id_customer'],
        'conversation_id': merged['tweet_id_customer'],
        'customer_message': merged['text_customer'],
        'brand_reply': merged['text_brand'],
    })
    
    # Basic cleaning
    output_df['customer_message_cleaned'] = output_df['customer_message'].apply(clean_text)
    # Filter out empty messages
    output_df = output_df[output_df['customer_message_cleaned'].str.len() > 10]
    
    os.makedirs('data', exist_ok=True)
    out_file = 'data/spotify_interactions.csv'
    output_df.to_csv(out_file, index=False)
    print(f"Saved {len(output_df)} interactions to {out_file}")

if __name__ == '__main__':
    main()
