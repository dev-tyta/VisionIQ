# importing libraries
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as fastercnn_model
from config import device


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
    


# to test model
ModelLoader.load_yolo(device)
ModelLoader.load_fastercnn(device)


#     # loop to construct bounding boxes on image.
#  
#             y = Y_1 - 15 if Y_1  over each object
#             y = Y_1 - 15 if Y_1 - 15 > 15 else Y_1 + 15

#             # adds the label text to the image.
#             cv2.putText(orig, label, (X_1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
#             cv2.putText(orig, f"Number of People: {people}", (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

#     return orig


# # function to perform object detection in videos
# def video_detection(video_path):
#     video = cv2.VideoCapture(video_path)
#     # frame_width = video.get(3)
#     # frame_height = video.get(4)

#     # out = cv2.VideoWriter(vid_out, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

#     while video.isOpened():
#         ret, frame = video.read()
#         vid = frame.copy()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = transforms.functional.to_tensor(frame)
#         frame = frame.to(device)
#         vid_detect = model([frame])[0]

#         for i in range(0, len(vid_detect["boxes"])):
#             confidence = vid_detect["scores"][i]

#             if confidence > 0.75:
#                 idx = int(vid_detect["labels"][i])
#                 box = vid_detect["boxes"][i].detach().cpu().numpy()
#                 (X_1, Y_1, X_2, Y_2) = box.astype("int")

#                 label = f"{classes[idx]}, {idx}: {confidence* 100}%"
#                 print(f"[INFO] {label}")

#                 cv2.rectangle(vid, (X_1, Y_1),
#                               (X_2, Y_2), colors[idx], 2)
#                 y = Y_1 - 15 if Y_1 - 15 > 15 else Y_1 + 15

#                 cv2.putText(vid, label, (X_1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

#     return vid
