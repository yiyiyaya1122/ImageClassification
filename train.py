import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from utils.trainer import Trainer
from utils.parse_args import parse_args
from utils.utils import *
from models.vit import ViT
from models.resnet import ResNet

def main():
    args = parse_args()
    logger = init_log(args)
    logger.info(args)

    start_time = time.time()
    
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 预期的输入尺寸是 224x224
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 均值和标准差
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(root='./data/Vegetable Images/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='./data/Vegetable Images/validation', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    # model = ResNet()
    model = ViT(num_classes=args.num_classes)
    
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 创建训练器对象并训练
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, epochs=args.epochs)
    trainer.train()

    # 记录 训练 & 验证 时间
    end_time = time.time()
    formatted_time = format_elapsed_time(end_time - start_time)
    logger.info(f"Train & val time: {formatted_time}.")

    # 测试
    if args.test:
        from utils.tester import Tester

        test_dataset = datasets.ImageFolder(root='./data/Vegetable Images/test', transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

        tester = Tester(model, test_loader, device)
        test_acc, precision, recall, f1 = tester.test()
        logger.info(f"Accuracy on test set: {test_acc * 100:.2f}%")




if __name__ == "__main__":
    main()

    
    


            
            
        


