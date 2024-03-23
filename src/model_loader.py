# importing modules
import cv2
import torch
from torchvision import transforms
from torchvision.models import detection
import numpy as np



# calling the Faster RCNN ResNet50 model
model = detection.fasterrcnn_resnet50_fpn_v2(pretrained=True, progress=True, pretrained_backbone=True).to(device)
print(model.eval())  # prints out the architecture of the model



# function to carry out object detection on images.
def img_detect(img_path):
    image = cv2.imread(img_path)  # reads the model using OpenCV
    image = cv2.resize(image, (640, 480))
    orig = image.copy()

    # changing the colorspace from BGR to RGB (since Pytorch trains only RGB image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.transpose((2, 0, 1))  # swapping the color channels from channels last to channels first

    image = np.expand_dims(image, axis=0)  # add batch dimension to the image
    image = image / 255.0  # scaling image from (0,255) to (0,1)
    image = torch.FloatTensor(image)  # changes the numpy array to a tensor.

    image = image.to(device)
    detections = model(image)[0]  # the image is passed to the model to get the bounding boxes

    people = 0
    # loop to construct bounding boxes on image.
    for i in range(0, len(detections["boxes"])):
        confidence = detections["scores"][i]  # get confidence score of each object in the image
        idx = int(detections["labels"][i])  # identifying the id of each of the classes in the image
        box = detections["boxes"][i].detach().cpu().numpy()  # gets the coordinates for the bounding boxes
        (X_1, Y_1, X_2, Y_2) = box.astype("int")

        if confidence > 0.75 and idx == 1:
            # matching the label index with its classes and its probability
            label = f"{classes[idx]}, {idx}: {confidence* 100}%"
            print(f"[INFO] {label}")
            people += 1

            cv2.rectangle(orig, (X_1, Y_1), (X_2, Y_2), colors[idx], 2)
            # draw bounding boxes over each object
            y = Y_1 - 15 if Y_1 - 15 > 15 else Y_1 + 15

            # adds the label text to the image.
            cv2.putText(orig, label, (X_1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
            cv2.putText(orig, f"Number of People: {people}", (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    return orig


# function to perform object detection in videos
def video_detection(video_path):
    video = cv2.VideoCapture(video_path)
    # frame_width = video.get(3)
    # frame_height = video.get(4)

    # out = cv2.VideoWriter(vid_out, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while video.isOpened():
        ret, frame = video.read()
        vid = frame.copy()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transforms.functional.to_tensor(frame)
        frame = frame.to(device)
        vid_detect = model([frame])[0]

        for i in range(0, len(vid_detect["boxes"])):
            confidence = vid_detect["scores"][i]

            if confidence > 0.75:
                idx = int(vid_detect["labels"][i])
                box = vid_detect["boxes"][i].detach().cpu().numpy()
                (X_1, Y_1, X_2, Y_2) = box.astype("int")

                label = f"{classes[idx]}, {idx}: {confidence* 100}%"
                print(f"[INFO] {label}")

                cv2.rectangle(vid, (X_1, Y_1),
                              (X_2, Y_2), colors[idx], 2)
                y = Y_1 - 15 if Y_1 - 15 > 15 else Y_1 + 15

                cv2.putText(vid, label, (X_1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    return vid
