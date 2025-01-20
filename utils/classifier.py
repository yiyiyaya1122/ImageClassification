import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from utils.utils import cvtColor
from models.resnet import ResNet

transform = transforms.Compose([
            transforms.Resize(224),  # Resize为目标大小
            transforms.ToTensor(),  # 将图片转换为Tensor并归一化到[0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 使用ImageNet的均值和标准差进行归一化
        ])

class Classifier:
        _defaults = {
        "model_path"        : './ckpts/resnet101_vegetable.pth',
        "classes_path"      : './data/Vegetable Images/classname.txt',
        "input_shape"       : [224, 224],
        "backbone"          : 'resnet101',
        "letterbox_image"   : False,
        "device"            : "cuda"
    }
        @classmethod
        def get_defaults(cls, key):
            if key in cls._defaults:
                return cls._defaults[key]
            else:
                return "Unrecognized attribute name '" + key + "'"
            
        def __init__(self, **kwargs):
            self.__dict__.update(self._defaults)

            for name, value in kwargs.items():
                setattr(self, name, value)

            self.generate()

        def generate(self):
            self.model = ResNet()
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.to(self.device)

        def detect_image(self, image, transform=transform):

            image = cvtColor(image)
            photo = transform(image).unsqueeze(0)  # 增加batch维度 (C, H, W) -> (1, C, H, W)

            self.model.eval()
            with torch.no_grad():                
                photo = photo.to(self.device)
                preds = torch.softmax(self.model(photo), dim=-1).cpu().numpy()[0]

            # class_name = self.class_names[np.argmax(preds)]
            class_name = np.argmax(preds)
            probability = np.max(preds)

            plt.subplot(1, 1, 1)
            plt.imshow(np.array(image))
            plt.title('Class:%s Probability:%.3f' %(class_name, probability))
            plt.show()
          
            return class_name, probability
