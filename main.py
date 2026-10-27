import os
import uvicorn
from fastapi import FastAPI
from scalar_fastapi import add_scalar_reference
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from dotenv import load_dotenv

# 1. Load variables from .env file
load_dotenv()

app = FastAPI(
    title="FastAPI with Pydantic AI & Scalar",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

add_scalar_reference(app, route="/docs")


class AgentRequest(BaseModel):
    prompt: str


@app.get("/", summary="Hello World Endpoint")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/health", summary="Health Check Endpoint")
def health_check():
    return {
        "status": "healthy",
        "message": "Service is up and running"
    }


@app.post("/basic-agent", summary="Predict Endpoint")
async def basic_agent(body: AgentRequest):
    """Returns a response using an OpenRouter model."""
    try:
        # Check if API key exists in environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return {"error": "OPENROUTER_API_KEY is not set in .env file"}

        # Specify model using OpenRouterModel explicitly
        model = OpenRouterModel("meta-llama/llama-3.1-8b-instruct")

        agent = Agent(
            model=model,
            instructions="You are a helpful assistant. Please answer concisely."
        )

        result = await agent.run(body.prompt)
        return {
            "response": result.output
        }
    except Exception as e:
        return {
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)