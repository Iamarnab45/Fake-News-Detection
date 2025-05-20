from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any
import uvicorn
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path

from utils.logger import app_logger
from utils.nlp_features import NLPFeatures
from models.fake_news_model import FakeNewsModel

# Get the current directory (where main.py is located)
BASE_DIR = Path(__file__).resolve().parent
print(f"BASE_DIR: {BASE_DIR}") # Diagnostic print

app = FastAPI(
    title="Fake News Detection System",
    description="API for detecting fake news articles using machine learning",
    version="1.0.0"
)

# Mount static files
STATIC_FILES_DIR = BASE_DIR / "static"
print(f"Serving static files from: {STATIC_FILES_DIR}") # Diagnostic print
app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

# Templates
TEMPLATES_DIR = BASE_DIR / "templates"
print(f"Serving templates from: {TEMPLATES_DIR}") # Diagnostic print
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
nlp_features = NLPFeatures()
model = FakeNewsModel(base_dir=BASE_DIR)

class NewsArticle(BaseModel):
    title: str = Field(..., min_length=10, max_length=500)
    content: str = Field(..., min_length=50, max_length=10000)
    source: Optional[str] = Field(None, max_length=200)

class URLRequest(BaseModel):
    url: HttpUrl

class PredictionResponse(BaseModel):
    is_fake: bool
    confidence: float
    explanation: str
    features: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str

def extract_article_content(url: str) -> tuple[str, str, str]:
    """
    Extract title and content from a news article URL.
    
    Args:
        url: URL of the news article
        
    Returns:
        Tuple of (title, content, source)
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.find('title').text.strip()
        
        # Extract content (this is a basic implementation, might need to be customized for different news sites)
        content = ' '.join([p.text for p in soup.find_all('p')])
        
        # Extract source from URL
        source = re.search(r'https?://(?:www\.)?([^/]+)', url).group(1)
        
        return title, content, source
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract content from URL: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions"""
    app_logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_url(request: URLRequest):
    """
    Analyze a news article from a URL.
    
    Args:
        request: URLRequest object containing the article URL
        
    Returns:
        PredictionResponse with analysis results
    """
    try:
        # Extract content from URL
        title, content, source = extract_article_content(str(request.url))
        
        # Combine title and content for analysis
        full_text = f"{title}\n{content}"
        
        # Extract NLP features
        features = nlp_features.extract_all_features(full_text)
        
        # Get prediction
        is_fake, confidence, explanation = model.predict(full_text)
        
        app_logger.info(
            f"Prediction made for article from {source}: "
            f"is_fake={is_fake}, confidence={confidence:.2f}"
        )
        
        return {
            "is_fake": is_fake,
            "confidence": confidence,
            "explanation": explanation,
            "features": features
        }
        
    except Exception as e:
        app_logger.error(f"Error processing URL: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing URL: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model.is_trained
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 