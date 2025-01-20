import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from utils.trainer import Trainer
from models.vit import ViT
from models.resnet import ResNet

def main(*args, **kwargs):
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 预期的输入尺寸是 224x224
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 均值和标准差
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(root='./data/Vegetable Images/train', transform=transform)
    test_dataset = datasets.ImageFolder(root='./data/Vegetable Images/validation', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的 ResNet 模型
    # model = ViT()
    model = ResNet()

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # 创建训练器对象并训练
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, device, epochs=5)
    trainer.train()



if __name__ == "__main__":
    main()

    
    


            
            
        


