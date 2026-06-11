# SHL Assessment Recommendation System

## Overview
This project recommends SHL assessments based on natural language job requirements.

Example:
Input:
"Backend Java Engineer with AWS experience"

Output:
Top 10 SHL assessments ranked using semantic similarity.

## Tech Stack
- Python
- FastAPI
- Sentence Transformers
- FAISS
- HuggingFace
- SHL Assessment Catalog

## Project Structure

app/
data/
evals/
tests/

## Installation

pip install -r requirements.txt

## Generate Embeddings

python app/embeddings.py

## Run API

python -m uvicorn app.main:app --reload

## Open Swagger

http://127.0.0.1:8000/docsREADME