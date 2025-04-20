from sentence_transformers import SentenceTransformer

class Embedder:
    """Lightweight embedding model."""
    def __init__(self, model_name: str = 'paraphrase-MiniLM-L3-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
