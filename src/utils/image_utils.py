import cv2
from src.model_setup.config import image_resize, device
import numpy as np
import torch


class ImageUtils:
    def __init__(self):
        self.image_resize = image_resize
        self.device = device


    def image_handling(self, image):
        image = cv2.imread(filename=image)  # reading image with cv2
        if image is None:
            print(f"Failed to load image at {image}")
        else:
            image = cv2.resize(image, dsize=self.image_resize)  # resizig the image to standard resolution
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # changing color from BGR to RGB

            image = image.transpose((2, 0, 1))  # swapping the color channels from channels last to channels first 

            image = np.expand_dims(image, axis=0)  # add batch dimension to the image
            image = image / 255.0  # scaling image from (0,255) to (0,1)
            image = torch.FloatTensor(image)  # changes the numpy array to a tensor

            image = image.to(self.device)

            return image  # return the image
        