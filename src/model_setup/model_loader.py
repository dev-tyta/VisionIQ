import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as fastercnn_model
from ultralytics import YOLO

from src.model_setup.config import device

class ModelLoader:
    def __init__(self):
        self.device = device

    def load_fastercnn(self):
        try:
            model = fastercnn_model(pretrained=True, progress=True, pretrained_backbone=True)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Failed to load Faster R-CNN model: {e}")
            return None

    def load_yolo(self, model_name="yolo11n"):
        try:
            yolo_model = YOLO(model_name, task="detection")
            print("YOLO Model Loaded")
            yolo_model.to(self.device)
            yolo_model.eval()
            return yolo_model
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return None


model = ModelLoader()
faster_rcnn_model = model.load_fastercnn()
yolo_model = model.load_yolo()

# Use some print statements or logging to check the models loaded
print("Faster R-CNN Model Loaded:", faster_rcnn_model is not None)
print("YOLO Model Loaded:", yolo_model is not None)