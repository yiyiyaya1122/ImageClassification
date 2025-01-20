import argparse
import os
import time

# === 设置环境变量  ===
# 忽略 mindspore 的 warnings
os.environ["GLOG_v"] = "4"
LD_LIBRARY_PATH = os.environ["LD_LIBRARY_PATH"]
# 根据你自己的环境进行设置
LD_LIBRARY_PATH = f"LD_LIBRARY_PATH=~/miniconda3/envs/mindspore/lib:{LD_LIBRARY_PATH}"
os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
# === === === === ===

# import mindspore as ms
# import mindspore.nn as nn
# from mindspore.nn.learning_rate_schedule import CosineDecayLR
# from mindspore.train import Model
# from mindspore.train.callback import CheckpointConfig

from utils.utils_t import *
from models.vit import *


def parse_args():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--lr_decay", default=1e-2, type=float)
    parser.add_argument("--num-classes", default=15, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--log-interval", default=50, type=int)
    parser.add_argument(
        "--resume", default="", type=str, help="测试时指定加载的 checkpoint"
    )
    parser.add_argument("--test", action="store_true", help="训练完成后执行测试")
    parser.add_argument("--device", default="GPU", choices=["CPU", "GPU"])

    # 模型参数
    parser.add_argument(
        "--embed-dims", default=384, type=int, help="经过卷积层后的通道数"
    )
    parser.add_argument("--patch-size", default=16, type=int, help="每个 patch 的大小")
    parser.add_argument("--depth", default=6, type=int, help="Transformer 的层数")
    parser.add_argument(
        "--mlp-ratio",
        default=4.0,
        type=float,
        help="MLP 中隐藏层通道数相比对 embed_dims 的比例",
    )
    parser.add_argument("--num-heads", default=16, type=int, help="多头注意力数量")
    # 注意保证 input_size 是 patch_size 的整数倍
    parser.add_argument("--input-size", default=224, type=int, help="缩放后的图片尺寸")

    args = parser.parse_args()
    return args


def trainval(args, logger):
    num_epochs = args.epochs
    lr = args.lr

    # 初始化数据集
    train_dataset = create_dataset("data/train", batch_size=args.bs, shuffle=True)
    val_dataset = create_dataset("data/validation", batch_size=args.bs)
    steps_per_epoch = train_dataset.get_dataset_size()

    # 创建模型
    model = ViT(
        embed_dims=args.embed_dims,
        patch_size=args.patch_size,
        depth=args.depth,
        num_classes=args.num_classes,
        mlp_ratio=args.mlp_ratio,
        num_heads=args.num_heads,
        image_size=args.input_size,
    )

    # 定义损失函数和优化器
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # 余弦退火
    cosine_decay_lr = CosineDecayLR(
        min_lr=args.lr_decay * lr,
        max_lr=lr,
        decay_steps=num_epochs * steps_per_epoch,
    )
    # 定义优化器
    optimizer = nn.Adam(model.trainable_params(), learning_rate=cosine_decay_lr)

    net = Model(model, loss_fn, optimizer, metrics={"accuracy"})

    # 保存 checkpoint
    config_ck = CheckpointConfig(
        save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=10
    )
    ckpt_cb = CustomModelCheckpoint(
        prefix="vit_custom", directory="./checkpoints", config=config_ck
    )

    # 日志及验证
    log_cb = LogCallback(
        net=net,
        logger=logger,
        eval_dataset=val_dataset,
        total_train_steps=train_dataset.get_dataset_size(),
        log_interval=args.log_interval,
    )

    # 训练模型
    net.train(
        epoch=num_epochs,
        train_dataset=train_dataset,
        callbacks=[ckpt_cb, log_cb],
    )

    if args.test:
        test_dataset = create_dataset("data/test", batch_size=args.bs)
        accuracy = custom_test(model, test_dataset)

        logger.info(f"Accuracy on test set: {accuracy*100:.4f}%")


def test(args, logger):
    test_dataset = create_dataset("data/test", batch_size=args.bs)
    model = ViT(
        embed_dims=args.embed_dims,
        patch_size=args.patch_size,
        depth=args.depth,
        num_classes=args.num_classes,
        mlp_ratio=args.mlp_ratio,
        num_heads=args.num_heads,
        image_size=args.input_size,
    )

    param_dict = ms.load_checkpoint(args.resume)
    ms.load_param_into_net(model, param_dict)
    print(f"Loaded checkpoint from {args.resume}")
    accuracy = custom_test(model, test_dataset)
    logger.info(f"Accuracy on test set: {accuracy*100:.4f}%")


if __name__ == "__main__":

    args = parse_args()

    logger = init_log(args)
    logger.info(args)

    ms.set_context(device_target=args.device)

    if args.resume:
        test(args, logger)
    else:
        start_time = time.time()
        trainval(args, logger)

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        if hours > 0:
            formatted_time = f"{hours}h {minutes}m {seconds}s"
        else:
            formatted_time = f"{minutes}m {seconds}s"

        logger.info(f"Train & val time: {formatted_time}.")
