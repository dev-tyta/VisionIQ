from model_loader import ModelLoader
from config import device, classes
from image_utils import ImageUtils
from video_utils import VideoUtils
import cv2
import torch

yolo_model = ModelLoader.load_yolo(device)
fasterrcnn_model = ModelLoader.load_fastercnn(device)
image_utils = ImageUtils()
video_utils = VideoUtils()

class Detections:
    def __init__(self):
        self.yolo_model = yolo_model
        self.fasterrcnn_model = fasterrcnn_model
        self.image_utils = image_utils
        self.vifeo_utils = video_utils
        self.device = device
        self.classes = classes

    def image_detection(self, image):
        def yolo_detection(self, image):
            image_handled = self.image_utils.image_handling(image)
            detections = self.yolo_model(image_handled)[0]
            return detections

        def fasterrcnn_detections(self, image):
            image_handled = self.image_utils.image_handling(image)
            detections = self.fasterrcnn_model(image_handled)
            return detections
        

    def video_detection(self, video_path):
        video = cv2.VideoCapture(video_path)
        frames = video_utils.process_video(video)
        for frame in frames:
            detections = self.yolo_model(frame)
            return detections
        