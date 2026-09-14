import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RetrievalSystem:
    def __init__(self, data_path='data/train_set.csv'):
        self.df = pd.read_csv(data_path)
        # Drop interactions with no replies
        self.df = self.df.dropna(subset=['customer_message_cleaned', 'brand_reply']).copy()
        
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['customer_message_cleaned'])
        
    def retrieve_similar(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(sims)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                'customer_message': row['customer_message'],
                'brand_reply': row['brand_reply'],
                'similarity': sims[idx]
            })
        return results
