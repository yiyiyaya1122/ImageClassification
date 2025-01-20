import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

class Tester:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.test_loader = test_loader
        self.acc_history = []  # 用于记录每个batch的准确率
        self.precision_history = []  # 用于记录每个batch的精确度
        self.recall_history = []  # 用于记录每个batch的召回率
        self.f1_history = []  # 用于记录每个batch的F1-score

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing", ncols=100, unit="batch")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = outputs.max(1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # 记录预测值和真实标签用于计算精度、召回率、F1-score
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 更新pbar信息
                test_acc = 100 * correct / total
                pbar.set_postfix(acc=test_acc)

        # 计算最终准确率
        test_acc = 100 * correct / total
        
        # 计算精度、召回率、F1-score
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # 记录每个batch的准确率、精确度、召回率、F1-score
        self.acc_history.append(test_acc)
        self.precision_history.append(precision)
        self.recall_history.append(recall)
        self.f1_history.append(f1)

        return test_acc, precision, recall, f1

    def plot_metrics(self):
        
        # 四个指标的名称
        labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        
        # 每个指标的值，假设它们存储在类的属性中
        scores = [self.acc_history[-1], self.precision_history[-1], self.recall_history[-1], self.f1_history[-1]]

        # 创建2x2的子图
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))  # 
        axs = axs.flatten()  # 将2D数组展平，方便后续的迭代

        # 绘制每个指标的水平柱状图
        for i, (label, score) in enumerate(zip(labels, scores)):
            axs[i].barh(label, score, color=['blue', 'green', 'red', 'orange'][i])  # 使用 barh 生成水平柱状图
            axs[i].set_xlabel('Score')  # 设置x轴标签
            axs[i].set_xlim(0, 1)  # 假设分数在0到1之间，设置x轴范围
            # axs[i].grid(True)

        # 调整布局，防止子图重叠
        plt.tight_layout()
        plt.show()

        

