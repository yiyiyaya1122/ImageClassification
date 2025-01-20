import torch
from torchvision.models.resnet import resnet101
import torch.nn as nn
import torchvision


def ResNet(cls_num=15, weights=torchvision.models.ResNet101_Weights.DEFAULT, freeze=True):

    model = resnet101(weights=weights)
    # 修改最后一层为我们任务的类别数
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cls_num)  

    if freeze:
        for param in model.parameters():
            param.requires_grad = False  # 冻结所有层
        for param in model.fc.parameters():
            param.requires_grad = True  # 解冻全连接层
            
    return model


    