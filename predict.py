import torch
import numpy as np
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from utils.classifier import Classifier
import os

def predict():
    classifier = Classifier()

    while True:
        img_path = input("please input image file path:")

        try:
            img = Image.open(img_path)

        except:
            print("cann't open image")
            continue

        else:
            class_name = classifier.detect_image(img)
            # print(class_name)

            
if __name__ == "__main__":
    predict()