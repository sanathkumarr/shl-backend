from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from models.model import Embedder
from utils.recommend import Recommender

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="RAG-based recommender for SHL product catalog",
    version="1.0.0"
)

# CORS – allow your Vercel‑hosted frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
      "https://your-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender
recommender = Recommender(
    csv_path="data/shl_assessments.csv",
    embedder=Embedder()
)

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

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    recs = recommender.recommend(req.query, top_k=10)
    if not recs:
        raise HTTPException(404, "No recommendations found.")
    return {"recommended_assessments": recs}
