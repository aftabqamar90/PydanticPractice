import os
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, Choices
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
from typesafe_sdk import AsyncTypeSafeClient

router = APIRouter()

_SAMPLE_REQUEST = {
    "text": (
        "Invoice #4421 for customer Sara Ahmed. Motor cover under ABC Takaful. "
        "Contribution paid. Shariah compliant policy for Toyota Corolla 2020."
    ),
    "instructions": (
        "Pick the single customer name from the choices that appears in the text."
    ),
    "items": ["Ali Khan", "Sara Ahmed", "John Doe"],
}


class TextAgentRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": _SAMPLE_REQUEST})

    text: str = Field(
        description="OCR dump, JSON blob, or any plain text to analyze.",
    )
    instructions: str = Field(
        description="What you want answered or decided from the text.",
    )
    items: list[str] = Field(
        min_length=1,
        description=(
            "Required candidates the model must choose from "
            "(e.g. customer names, conventional/takaful). "
            "Include a value like 'none' yourself if no-match is allowed."
        ),
    )


@router.post("/text-agent", summary="generic text agent endpoint")
async def text_agent(body: TextAgentRequest):
    """Choose one of `items` from OCR/JSON/text via TypeSafe Choices.

    Always pass candidate options in `items` (required). The model picks exactly
    one. For policy type, use items like `["conventional", "takaful"]`. For name
    match, pass the customer list. Add `"none"` to items if no-match is allowed.
    """
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return {"error": "OPENROUTER_API_KEY is not set in .env file"}

        client = AsyncTypeSafeClient(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL"),
        )
        model = TypeSafeModel(
            os.getenv("OPENROUTER_DECISION_MODEL"),
            provider=TypeSafeProvider(typesafe_client=client),
        )

        choices = {item: item for item in body.items}
        agent = Agent(model=model, instructions=body.instructions)
        result = await agent.run(
            body.text,
            output_type=Choices(choices, description=body.instructions),
        )
        print(result)
        response = {"answer": result.output}
        provider_details = getattr(result.response, "provider_details", None)
        if provider_details and "probabilities" in provider_details:
            response["probabilities"] = provider_details["probabilities"].get(
                "response", provider_details["probabilities"]
            )
        else:
            response["probabilities"] = None
        return response
    except Exception as e:
        return {"error": str(e)}
