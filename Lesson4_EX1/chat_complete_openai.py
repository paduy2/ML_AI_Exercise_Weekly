import os
import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You are an IT Developer assistant. Help debugging code in various programming languages (Python, JavaScript, Java, C++, etc.)."},
            {"role": "user", "content": f"{prompt}"}
         ]
    )
    return response.choices[0].message

# Endpoint for chatbot
@app.post("/chat/")
async def chat_endpoint(prompt: Prompt):
    return {"response": get_openai_response(prompt.prompt)}

# Handling OPTIONS requests explicitly
@app.options("/chat/")
async def options_endpoint(request: Request):
    return {"Allow": "POST"}  # Specify allowed methods for the endpoint

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI server with uvicorn
    uvicorn.run(app, host="localhost", port=8000)
