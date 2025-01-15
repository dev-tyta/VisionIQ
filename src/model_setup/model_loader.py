import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as fastercnn_model
from transformers import AutoTokenizer, AutoModelForPreTraining
import sys


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

    def load_yolo(self, model_name="yolov5s"):
        try:
            model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
            print("YOLO Model Loaded")
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return None


model = ModelLoader()
faster_rcnn_model = model.load_fastercnn()
yolo_model = model.load_yolo()

# Use some print statements or logging to check the models loaded
print("Faster R-CNN Model Loaded:", faster_rcnn_model is not None)
print("YOLO Model Loaded:", yolo_model is not None)