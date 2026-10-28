import uvicorn
from fastapi import FastAPI
from scalar_fastapi import add_scalar_reference
from dotenv import load_dotenv

from basic_agent import router as basic_agent_router
from decision_agent import router as decision_agent_router
from text_agent import router as text_agent_router
# 1. Load variables from .env file
load_dotenv()

app = FastAPI(
    title="FastAPI with Pydantic AI & Scalar",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

add_scalar_reference(app, route="/docs")
app.include_router(basic_agent_router)
app.include_router(decision_agent_router)
app.include_router(text_agent_router)


@app.get("/", summary="Hello World Endpoint")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/health", summary="Health Check Endpoint")
def health_check():
    return {
        "status": "healthy",
        "message": "Service is up and running"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
