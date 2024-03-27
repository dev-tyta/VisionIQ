from model_loader import ModelLoader
from config import device, classes, model_confidence
from image_utils import ImageUtils
from video_utils import VideoUtils
import cv2

yolo_model = ModelLoader.load_yolo(device)
fasterrcnn_model = ModelLoader.load_fastercnn(device)
image_utils = ImageUtils()
video_utils = VideoUtils()

class Detections:
    def __init__(self):
        self.yolo_model = yolo_model
        self.fasterrcnn_model = fasterrcnn_model
        self.image_utils = image_utils
        self.video_utils = video_utils
        self.device = device
        self.classes = classes
        self.model_confidence = model_confidence

    def image_detection(self, image):
        image_handled = self.image_utils.image_handling(image)
        detections = self.yolo_model(image_handled)[0]
        return detections

    def video_detection(self, video_path):
        video = cv2.VideoCapture(video_path)
        frames = video_utils.process_video(video)
        people_count_per_frame = []
        for frame in frames:
            detections = self.yolo_model(frame)[0]
            people_count = self.people_count(detections)
            people_count_per_frame.append(people_count)
        
        return people_count_per_frame
        

    def people_count(self, detections):
        people = 0
        for i in range(0, len(detections["boxes"])):
            confidence = detections["scores"][i]
            class_idx = int(detections["labels"][i])

            if confidence > self.model_confidence and class_idx == 1:
                label = f"{self.classes[class_idx]}, {class_idx}: {confidence* 100}%"
                print(f"[INFO] {label}")
                people += 1
