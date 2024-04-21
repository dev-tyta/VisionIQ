# making a test for the project
from src.model.fasterrcnn_detections import Detections

image_path = 'data/acc.jpg'
faster_rcnn = Detections()

output = faster_rcnn.image_detection(image_path)
counting = faster_rcnn.people_count(output)

print(f"[INFO] People in the image: {counting}")