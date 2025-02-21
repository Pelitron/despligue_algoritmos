from fastapi import FastAPI
from endpoints import router as api_router

app = FastAPI(title="FastAPI con Hugging Face")

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
