import torch
import torch.nn as nn
from tqdm import tqdm
import mlflow
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, epochs=10, lr=0.001):
        """
        通用训练器，用于训练和验证深度学习模型。
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def train_one_epoch(self):
        """
        单轮训练。
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", ncols=100, unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        """
        验证模型。
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating", ncols=100, unit="batch")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = 100 * correct / total
        return val_loss, val_acc

    def train(self):
        """
        训练与验证。
        """
        with mlflow.start_run():
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("learning_rate", self.lr)

            for epoch in range(1, self.epochs + 1):
                print(f"Epoch {epoch}/{self.epochs}")

                # 训练阶段
                train_loss, train_acc = self.train_one_epoch()
                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)

                # 验证阶段
                val_loss, val_acc = self.validate()
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)

                if epoch % 1 == 0:
                    self.save_model(f"./ckpts/epoch{epoch}.pth")

                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            self.plot_metrics()
    
    def save_model(self, save_path="./ckpts/model.pth"):
        torch.save(self.model.state_dict(), save_path)
        

    def plot_metrics(self):
        """
        绘制训练和验证的损失与准确率曲线。
        """
        epochs = range(1, self.epochs + 1)
        train_loss = self.history["train_loss"]
        val_loss = self.history["val_loss"]
        train_acc = self.history["train_acc"]
        val_acc = self.history["val_acc"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 绘制损失
        ax1.plot(epochs, train_loss, label="Train Loss", color="blue", linestyle="-", linewidth=2)
        ax1.plot(epochs, val_loss, label="Validation Loss", color="orange", linestyle="--", linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Loss Curve", fontsize=14)
        ax1.legend()
        ax1.grid()

        # 绘制准确率
        ax2.plot(epochs, train_acc, label="Train Accuracy", color="green", linestyle="-", linewidth=2)
        ax2.plot(epochs, val_acc, label="Validation Accuracy", color="red", linestyle="--", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy (%)", fontsize=12)
        ax2.set_title("Accuracy Curve", fontsize=14)
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        plt.show()

