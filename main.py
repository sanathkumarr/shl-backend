from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from models.model import Embedder
from utils.recommend import Recommender
from typing import Optional
# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="RAG-based recommender for SHL product catalog",
    version="1.0.0"
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Allow requests from your React frontend
origins = [
    "https://shl-frontend-delta.vercel.app/",
    "https://shl-frontend-sanath-kumars-projects-bb34292c.vercel.app/",
    "https://shl-frontend-git-main-sanath-kumars-projects-bb34292c.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Origins allowed
    allow_credentials=True,
    allow_methods=["*"],              # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],              # Allow all headers
)

# your existing routes and logic

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)