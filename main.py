import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vit import ViT  # 假设你有 ViT 的 PyTorch 实现
from utils.utils import *
from utils.arg_parse import parse_args


def trainval(args, logger):
    num_epochs = args.epochs
    lr = args.lr

    # 初始化数据集
    train_dataset = create_dataset("./data/Vegetable Images/train", batch_size=args.bs, shuffle=True)
    val_dataset = create_dataset("./data/Vegetable Images/validation", batch_size=args.bs)
    steps_per_epoch = len(train_dataset)

    # 创建模型
    model = ViT(
        embed_dims=args.embed_dims,
        patch_size=args.patch_size,
        depth=args.depth,
        num_classes=args.num_classes,
        mlp_ratio=args.mlp_ratio,
        num_heads=args.num_heads,
        image_size=args.input_size,
    ).to(args.device)

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 余弦退火学习率调度
    cosine_decay_lr = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * steps_per_epoch, eta_min=args.lr_decay * lr
    )

    # 日志和检查点回调
    log_cb = LogCallback(model, logger, val_dataset, len(train_dataset), args.log_interval)
    checkpoint_cb = CustomModelCheckpoint(
        model, save_dir="./checkpoints", save_freq=steps_per_epoch
    )

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_dataset):
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            optimizer.zero_grad()  # 清除梯度
            outputs = model(inputs)  # 前向传播
            loss = loss_fn(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 每 log_interval 步打印一次
            if (batch_idx + 1) % args.log_interval == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{steps_per_epoch}], Loss: {running_loss / (batch_idx+1):.4f}, Accuracy: {100 * correct / total:.2f}%")

        # 更新学习率
        cosine_decay_lr.step()

        logger.info(f"Epoch [{epoch+1}/{num_epochs}] complete. Average loss: {running_loss / steps_per_epoch:.4f}, Accuracy: {100 * correct / total:.2f}%")

        # 验证阶段
        log_cb(epoch)

        # 保存模型检查点
        checkpoint_cb(epoch)

    if args.test:
        test(args, logger, model)


def test(args, logger, model=None):
    test_dataset = create_dataset("data/Vegetable Images/test", batch_size=args.bs)
    
    if model is None:
        model = ViT(
            embed_dims=args.embed_dims,
            patch_size=args.patch_size,
            depth=args.depth,
            num_classes=args.num_classes,
            mlp_ratio=args.mlp_ratio,
            num_heads=args.num_heads,
            image_size=args.input_size,
        ).to(args.device)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {args.resume}")

    accuracy = custom_test(model, test_dataset, args.device)
    logger.info(f"Accuracy on test set: {accuracy*100:.4f}%")


if __name__ == "__main__":
    args = parse_args()

    # 初始化日志
    logger = init_log(args)
    # logger = logging.getLogger()
    logger.info("1")
    # 设置设备
    device = torch.device(args.device)
    
    # if args.resume:
    #     test(args, logger)
    # else:
    #     start_time = time.time()
    #     trainval(args, logger)

    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     hours = int(elapsed_time // 3600)
    #     minutes = int((elapsed_time % 3600) // 60)
    #     seconds = int(elapsed_time % 60)
    #     if hours > 0:
    #         formatted_time = f"{hours}h {minutes}m {seconds}s"
    #     else:
    #         formatted_time = f"{minutes}m {seconds}s"

    #     logger.info(f"Train & val time: {formatted_time}.")
