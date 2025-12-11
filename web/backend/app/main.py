from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import rankings
from app.api.model import explain
import os
import re

app = FastAPI(
    title="CFB Rating Model API",
    description="API for college football team ratings and rankings",
    version="1.0.0"
)

# Get frontend URL from environment variable (for production)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Normalize the URL (remove trailing slash if present)
if FRONTEND_URL.endswith("/"):
    FRONTEND_URL = FRONTEND_URL.rstrip("/")

# Configure CORS to allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        FRONTEND_URL,  # Production frontend URL from env var
        "https://cfb-model-project.vercel.app",  # Vercel production URL
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",  # Allow all Vercel preview deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rankings.router, prefix="/api/rankings", tags=["rankings"])
app.include_router(explain.router, prefix="/api/model", tags=["model"])

@app.get("/")
async def root():
    return {
        "message": "CFB Rating Model API",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}