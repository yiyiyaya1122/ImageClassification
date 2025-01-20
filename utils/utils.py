import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np

def custom_test(model: nn.Module, dataset: DataLoader, device: torch.device) -> float:
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted_classes = torch.max(outputs, dim=-1)

            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    return accuracy


class CustomModelCheckpoint:
    def __init__(self, prefix, directory):
        self.prefix = prefix
        self.directory = directory

    def save_checkpoint(self, model, epoch_num):
        os.makedirs(self.directory, exist_ok=True)
        checkpoint_path = os.path.join(self.directory, f"{self.prefix}_e{epoch_num}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


class LogCallback:
    def __init__(self, model, logger, eval_dataset, total_train_steps, log_interval=10):
        self.model = model
        self.logger = logger
        self.eval_dataset = eval_dataset
        self.log_interval = log_interval
        self.step_count = 0
        self.total_train_steps = total_train_steps
        self.losses = []

    def log_train_step(self, epoch_num, step_count, loss):
        self.step_count += 1
        self.losses.append(loss.item())
        if self.step_count % self.log_interval == 0:
            avg_loss = sum(self.losses) / len(self.losses)
            self.losses = []
            print(f"Epoch #{epoch_num}, Step [{step_count}/{self.total_train_steps}], Loss: {avg_loss:.6f}")

    def log_epoch_end(self, epoch_num):
        accuracy = custom_test(self.model, self.eval_dataset, device)
        self.logger.info(f"Epoch {epoch_num}, Accuracy on val set: {accuracy*100:.4f}%")


def create_dataset(data_path, batch_size=32, image_size=224, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader


def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)


def init_log(args):
    now = datetime.now()
    formatted_time = now.strftime("%m-%d_%H-%M-%S")
    mkdir_p("logs")
    log_path = os.path.join(
        "logs",
        f"bs{args.bs}_d{args.depth}_E{args.embed_dims}_{args.epochs}e_{formatted_time}",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # formatter = logging.Formatter("%(message)s")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")


    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 