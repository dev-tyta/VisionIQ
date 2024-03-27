# making a test for the project
from detections import Detections

image_path = '../data/fam1.HEIC'

output = Detections.image_detection(image_path)
counting = Detections.batch_people_count(output)