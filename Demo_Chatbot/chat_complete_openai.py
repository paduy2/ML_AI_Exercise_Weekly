import os
import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

# Set up your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

app = FastAPI()

# CORS (Cross-Origin Resource Sharing) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your front-end's URL if possible
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow these methods
    allow_headers=["*"],  # Allow all headers
)

class Prompt(BaseModel):
    prompt: str

def get_openai_response(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are an IT Developer assistant. Help debugging code in various programming languages (Python, JavaScript, Java, C++, etc.)."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for chatbot
@app.post("/chat/")
async def chat_endpoint(prompt: Prompt):
    try:
        response = get_openai_response(prompt.prompt)
        return {"response": response}
    except HTTPException as e:
        return {"error": str(e.detail)}
    except Exception as e:
        return {"error": str(e)}

# Handling OPTIONS requests explicitly
@app.options("/chat/")
async def options_endpoint(request: Request):
    return {"Allow": "POST"}  # Specify allowed methods for the endpoint

if __name__ == "__main__":
    import uvicorn
    import sys
    import asyncio
    # Use Selector event loop on Windows to avoid Proactor issues
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Paths to the SSL certificate and key files
    ssl_certfile = "cert.pem"
    ssl_keyfile = "key.pem"
    
    # Run the FastAPI server with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile
    )
