from fastapi import FastAPI, Body, File, UploadedFile, status, HTTPException, status
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import AnyHttpUrl, UrlConstraints
from config import settings


app = FastAPI(
    title=settings.PROJECT_NAME,
    description= "VisionIQ API",
    version="1.0.0",
    openapi_url="/openapi.json"
)

if settings:
    app.add_middleware(
        CORSMiddleware,
        allow_origins="*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/")
async def root():
    return {"message":"Welcome to VisionIQ!!"}


@app.get("/health")
def health():
    return {"message":"OK"}

