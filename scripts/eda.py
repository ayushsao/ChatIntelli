import pandas as pd
import kagglehub
import os

path = kagglehub.dataset_download('thoughtvector/customer-support-on-twitter')
csv_path = os.path.join(path, 'twcs', 'twcs.csv')

if not os.path.exists(csv_path):
    print("CSV not found at", csv_path)
    csv_path = os.path.join(path, 'twcs.csv') # Maybe it is not in twcs/? Wait, list_dir said twcs/twcs.csv

print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

print("Columns:", df.columns)
print("Data head:", df.head())

# Find the most frequent brands
author_counts = df['author_id'].value_counts()
print("\nTop authors (brands):")
print(author_counts.head(50))

# Profile the conversation structure
# The dataset has 'author_id', 'inbound' (True if customer to brand), 'text', 'tweet_id', 'in_response_to_tweet_id'
brands = df[df['inbound'] == False]['author_id'].value_counts().head(20)
print("\nTop brands (outbound tweets):")
print(brands)
