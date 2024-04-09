# importing libraries
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as fastercnn_model
from src.model.config import device


class ModelLoader:
    def __init__(self):
        self.device = device
        
    def load_fastercnn(device):
        # loading Faster RCNN ResNet50 model
        model = fastercnn_model(pretrained=True, progress=True, pretrained_backbone=True)
        model.to(device)  
        model.eval() # prints out the architecture of the model
        return model
    
    def load_yolo(device, model_name= "yolov5s"):
        model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True, trust_repo=True)
        model.to(device)
        model.eval()
        return model 
    