from fastapi import FastAPI, Query
from app.retriever import retrieve
from app.models import RecommendationResponse

app = FastAPI(
    title="SHL Assessment Recommendation API",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get(
    "/recommend",
    response_model=RecommendationResponse
)
def recommend(
    query: str = Query(...),
    top_k: int = 10
):
    results = retrieve(query, top_k)

    return RecommendationResponse.from_retriever(
        query=query,
        raw=results
    )
@app.get("/")
def home():
    return {
        "name": "SHL Assessment Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "recommend": "/recommend"
    }