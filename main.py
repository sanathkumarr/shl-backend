from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from models.model import Embedder
from utils.recommend import Recommender
import os

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="RAG-based recommender for SHL product catalog",
    version="1.0.0"
)

# ✅ Allow all origins for testing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate recommender on startup
recommender = Recommender(csv_path='data/shl_assessments.csv', embedder=Embedder())

# ---- Pydantic schemas ----
class RecommendRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: Optional[int] = None
    remote_support: str
    test_type: List[str]

class RecommendResponse(BaseModel):
    recommended_assessments: List[Assessment]

# ---- Endpoints ----
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    recs = recommender.recommend(req.query, top_k=10)
    if not recs:
        raise HTTPException(status_code=404, detail="No recommendations found for the given query.")
    return {"recommended_assessments": recs}

# ---- Run with dynamic port for Render ----
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
