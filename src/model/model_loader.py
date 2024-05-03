# importing libraries
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as fastercnn_model
from transformers import AutoProcessor, AutoModelForPreTraining
from config import device


class ModelLoader:
    def __init__(self):
        self.device = device

    def load_fastercnn(self):
        # loading Faster RCNN ResNet50 model
        model = fastercnn_model(
            pretrained=True, progress=True, pretrained_backbone=True)
        model.to(self.device)
        model.eval()  # prints out the architecture of the model
        return model

    def load_yolo(self, model_name="yolov5s"):
        model = torch.hub.load("ultralytics/yolov5",
                               model_name, pretrained=True, trust_repo=True)
        model.to(self.device)
        model.eval()
        return model

    def load_llava(self):
        llava_processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf")
        llava_model = AutoModelForPreTraining.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf")
        llava_model.to(self.device)
        llava_model.eval()
        return llava_processor, llava_model


model = ModelLoader()
model.load_llava()
