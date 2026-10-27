import os
from fastapi import APIRouter
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel

router = APIRouter()


class AgentRequest(BaseModel):
    prompt: str


@router.post("/basic-agent", summary="Predict Endpoint")
async def basic_agent(body: AgentRequest):
    """Returns a response using an OpenRouter model."""
    try:
        # Check if API key exists in environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return {"error": "OPENROUTER_API_KEY is not set in .env file"}

        # Specify model using OpenRouterModel explicitly
        model = OpenRouterModel(os.getenv("OPENROUTER_MODEL"))

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
