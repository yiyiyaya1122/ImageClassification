import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from utils.tester import Tester
from models.resnet import ResNet
from models.vit import ViT
import torch

def main(*args, **kwargs):
    pass

if __name__ == "__main__":
    # model = ResNet(weights=None)
    model = ViT()
    # print(model.fc.out_features)
    ckpt_path = './ckpts/vit_vegetable-2.pth'
    model.load_state_dict(torch.load(ckpt_path))

    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 预期的输入尺寸是 224x224
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 均值和标准差
    ])
    test_dataset = datasets.ImageFolder("C:/Users/12967/Desktop/ImageClassification/data/Vegetable Images/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tester = Tester(model, test_loader, device)
    test_acc, precision, recall, f1 = tester.test()

    print(f"Test Accuracy: {test_acc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # 绘制指标曲线
    tester.plot_metrics()