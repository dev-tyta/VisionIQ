from fastapi import APIRouter, status, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

from src.count.yolo_detections import YoloDetections
from src.count.fasterrcnn_detections import Detections
from src.model_setup.model_loader import ModelLoader

load_dotenv()

router = APIRouter(
    prefix="/detect",
    tags=["Detect People with Object Detection Models"],
)

model = ModelLoader()
yolo_model = YoloDetections()
frcnn_model = Detections()

@router.get("/health",  tags=["Health Check on Model Availability"])
def health_check():
    return {"status": "ok"}


@router.post("/",status_code=status.HTTP_200_OK)
def check_models():
    faster_rcnn_model = model.load_fastercnn()
    yolo_model = model.load_yolo()

    if faster_rcnn_model is not None and yolo_model is not None:
        return {"message": "Models Loaded Successfully"}
    else:
        return {"message": "Models Failed to Load"}


@router.post("/yolo")
def yolo_count(
    img: UploadFile = File(
        default=None,
        description="Takes in Image file for processing and counting outputs."
    )
):
    count = yolo_model.detect_with_yolo(img)

    return {"message": f"Total number of people in image: {count}"}  



@router.post("/frcnn")
def frcnn_count(
        img: UploadFile = File(
            default=None,
            description="Img File to process for counting using the FasterRCNN model."
        )
):
    count2 = frcnn_model.image_detection(img)

    return {"message": f"Total number of people in image: {count2}"}