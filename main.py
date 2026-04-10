from fastapi import FastAPI
from src.routers.predictions import router as predictions_router

app = FastAPI()

app.include_router(predictions_router)