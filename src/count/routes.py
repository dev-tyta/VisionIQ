from fastapi import APIRouter, status, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv


load_dotenv()

router = APIRouter(
    prefix="/count",
    tags=["Count Models"],
)


@router.get("/health",  tags=["Health Check on Model Availability"])
def health_check():
    return {"status": "ok"}


@router.post("/",status_code=status.HTTP_200_OK)
def check_model():
    pass

@router.post("/yolo")
def yolo_count():
    pass

@router.post("/frcnn")
def frcnn_count():
    pass

