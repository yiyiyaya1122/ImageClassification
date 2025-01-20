import torch
import torch.optim as optim
import torch.nn as nn

class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False, path='./ckpts/best.pth'):
        self.patience = patience  # 容忍多少个 epoch 没有改善
        self.delta = delta  # 最小变化量
        self.verbose = verbose  # 是否打印信息
        self.path = path  # 模型保存路径
        self.counter = 0  # 没有改善的 epoch 计数器
        self.best_loss = float('inf')  # 初始化为正无穷，表示初始时损失值无限大
        self.early_stop = False  # 是否提前停止

    def __call__(self, val_loss, model):
        # 如果是第一次调用，初始化最佳验证损失
        if val_loss < self.best_loss - self.delta:
            # self.best_loss = val_loss  # 更新最小验证损失
            self.save_checkpoint(val_loss, model)  # 保存当前模型
            self.counter = 0  # 重置计数器
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:  # 如果没有改善超过设定的 patience
                self.early_stop = True  # 触发提前停止

    def save_checkpoint(self, val_loss, model):

        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)  # 保存模型
        self.best_loss = val_loss  # 更新最佳损失



