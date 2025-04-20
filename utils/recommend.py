import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from models.model import Embedder

class Recommender:
    """RAG-based recommender for SHL assessments with enhanced ranking logic."""
    def __init__(self, csv_path: str, embedder: Embedder = None, alpha: float = 0.7):
        # Load data
        self.df = pd.read_csv(csv_path)
        # Combine name + description for richer embeddings
        self.df['text'] = self.df['name'] + '. ' + self.df['description'].fillna('')
        # Extract numeric duration
        self.df['duration_num'] = self.df['duration'].str.extract(r'(\d+)').astype(float).fillna(0)
        # Initialize embedder
        self.embedder = embedder or Embedder()
        # Precompute SBERT embeddings
        self.sbert_embeddings = self.embedder.embed(self.df['text'].tolist())
        # Precompute TF-IDF matrix
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['text'])
        # Weight for combining SBERT and TF-IDF similarities
        self.alpha = alpha

    def recommend(self, query: str, top_k: int = 10):
        """Return top_k recommendations for the given query with advanced ranking."""
        # Semantic similarity via SBERT
        query_emb = self.embedder.embed([query])[0]
        sbert_sims = cosine_similarity([query_emb], self.sbert_embeddings)[0]
        # Lexical similarity via TF-IDF
        query_tfidf = self.vectorizer.transform([query])
        tfidf_sims = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        # Combined similarity score
        sims = self.alpha * sbert_sims + (1 - self.alpha) * tfidf_sims

        # Parse desired max duration from query (if any)
        max_duration = None
        dur_match = re.search(r'(\d+)\s*(?:min|minute)', query.lower())
        if dur_match:
            max_duration = float(dur_match.group(1))

        results = []
        for idx, score in enumerate(sims):
            row = self.df.iloc[idx]
            duration = int(row['duration_num']) if row['duration_num'] else None
            # Penalize if duration exceeds desired max
            if max_duration is not None and duration is not None and duration > max_duration:
                penalty = max(0.0, 1 - ((duration - max_duration) / max_duration))
                score *= penalty

            results.append({
                "url": row['url'],
                "adaptive_support": row['adaptive_irt'],
                "description": row['description'],
                "duration": duration,
                "remote_support": row['remote_testing'],
                "test_type": [t.strip() for t in row['test_type'].split(',')],
                "score": float(score)
            })

        # Rank by score and select top_k
        ranked = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        # Drop score from final output
        for rec in ranked:
            rec.pop('score', None)

        return ranked