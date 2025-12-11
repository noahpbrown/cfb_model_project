from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

router = APIRouter()

# Don't initialize client here - do it lazily in the function

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# System prompt with context about the model
SYSTEM_PROMPT = """You are a helpful assistant for a College Football (CFB) Rating Model system. 
You help users understand:

1. **The Model**: This is an XGBoost machine learning model that predicts team strength ratings based on:
   - Offensive and defensive statistics (EPA, success rate, explosiveness)
   - Strength of schedule
   - Preseason priors (talent ratings, returning production)
   - Opponent-adjusted statistics
   - Resume features (wins, losses, margins)

2. **Rankings**: The model generates weekly Top 25 rankings based on predicted team strength ratings. 
   Rankings are calculated using only data available up to that week (no lookahead bias).

3. **Features**: The model uses 34 features including:
   - Cumulative offensive/defensive stats
   - Pass/rush splits
   - Strength of schedule metrics
   - Preseason priors
   - Opponent-adjusted metrics

4. **Predictions**: You can discuss how the model works, explain rankings, compare teams, and help users 
   understand college football statistics and analytics.

Be friendly, helpful, and accurate. If you don't know something specific about the model's implementation, 
it's okay to say so."""

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that uses OpenAI to answer questions about the CFB Rating Model.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        )
    
    # Initialize client here, only when needed
    openai_client = OpenAI(api_key=api_key)
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency, can upgrade to gpt-4o if needed
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        
        assistant_message = response.choices[0].message.content
        
        return ChatResponse(response=assistant_message)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )