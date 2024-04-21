# making a test for the project
from src.model.fasterrcnn_detections import Detections
from src.model.yolo_detections import YoloDetections

image_path = 'data/acc.jpg'
faster_rcnn = Detections()
yolo = YoloDetections

output = yolo.image_detection(image_path)
counting = yolo.people_count(output)

print(f"[INFO] People in the image: {counting}")