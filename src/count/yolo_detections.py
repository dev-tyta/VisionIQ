import torch
from src.model_setup.model_loader import ModelLoader
from src.model_setup.config import device, classes, model_confidence
from src.utils.image_utils import ImageUtils
from src.utils.video_utils import VideoUtils
import cv2


class YoloDetections:
    def __init__(self):
        self.image_utils = ImageUtils()
        self.video_utils = VideoUtils()
        self.device = device
        self.classes = classes
        self.model_confidence = model_confidence
        self.model = ModelLoader()
        self.yolo_model = self.model.load_yolo()

    def detect_with_yolo(self, image):
        if isinstance (image, str):
            detections = self.yolo_model(image)
            # print(detections[0])
            people = self.people_count(detections)

        else:
            detections = self.yolo_model(image)
            people = self.people_count(detections)

        return people

    def batch_image_detection(self, images):
        images_tensor = torch.stack(images).to(self.device)
        detections = self.yolo_model(images_tensor)

        all_processed_detections = []
        for detection in detections.xyxy:
            processed = self.process_yolo_detections(detection)
            all_processed_detections.append(processed)

        return all_processed_detections

    def video_detection(self, video_path):
        video = cv2.VideoCapture(video_path)
        frames = self.video_utils.process_video(video)
        frame_batches = self.video_utils.create_frame_batches(frames)

        all_detections = []
        for batch in frame_batches:
            batch_tensor = torch.stack(batch).to(self.device)
            detections = self.yolo_model(batch_tensor)[0]
            for det in detections.xyxy:
                all_detections.extend(det.cpu().numpy())
        return all_detections

    def process_yolo_detections(self, detections):
        # YOLO detections processing
        processed_detections = []
        for detection in detections.xyxy[0]:
            _, _, _, _, confidence, class_id = detection
            class_id = int(class_id)
            class_name = self.classes[class_id]
            if confidence.item() > self.model_confidence:
                processed_detections.append(
                    (class_id, class_name, confidence.item()))
        return processed_detections

    def people_count(self, detections):
        people = 0

        if isinstance(detections, list):
            for result in detections:
                boxes = result.boxes
                print(boxes)
            person_detections = [
                det for det in boxes
                if det.cls.item() == 0
            ]
            people = len(person_detections)
        else:
            person_detections = detections.boxes.cls == 0
            people = int(person_detections.sum())
        return people

    def batch_people_count(self, batch_detections):
        counts_per_batch = [self.people_count(
            detections) for detections in batch_detections]
        return counts_per_batch
