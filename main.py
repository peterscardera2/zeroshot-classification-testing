from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.item_classification_router import item_classification_router

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "localhost:8000",
    ],
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=["*"],
)

app.include_router(item_classification_router, prefix="/api")
