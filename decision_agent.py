import os
from fastapi import APIRouter
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
from typesafe_sdk import AsyncTypeSafeClient

router = APIRouter()


class AgentRequest(BaseModel):
    question: str


@router.post("/decision-agent", summary="decision agent endpoint")
async def decision_agent(body: AgentRequest):
    """Returns a response using Jev via OpenRouter System One."""
    try:
        # Check if API key exists in environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return {"error": "OPENROUTER_API_KEY is not set in .env file"}

        # Route TypeSafe/Jev through OpenRouter System One, not TypeSafe directly
        client = AsyncTypeSafeClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api",
        )
        model = TypeSafeModel(
            "jev-latest",
            provider=TypeSafeProvider(typesafe_client=client),
        )

        agent = Agent(
            model=model,
            instructions="Is this request harmful?",
            output_type=bool,
        )

        result = await agent.run(body.question)
        print(result)
        return {
            "response": result.output
        }
    except Exception as e:
        return {
            "error": str(e)
        }
