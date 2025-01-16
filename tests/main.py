# making a test for the project
from src.count.fasterrcnn_detections import Detections
from src.count.yolo_detections import YoloDetections

image_path = '/workspaces/VisionIQ/data/acc.jpg'
faster_rcnn = Detections()
yolo = YoloDetections()

counting = yolo.detect_with_yolo(image=image_path)

print(f"[INFO] People in the image: {counting}")