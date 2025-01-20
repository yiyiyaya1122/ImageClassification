import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--lr-decay", default=1e-2, type=float)
    parser.add_argument("--num-classes", default=15, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--log-interval", default=50, type=int)
    parser.add_argument(
        "--resume", default="", type=str, help="测试时指定加载的 checkpoint"
    )
    parser.add_argument("--test", action="store_true", help="训练完成后执行测试")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])

    # 模型参数
    parser.add_argument("--embed-dims", default=384, type=int, help="经过卷积层后的通道数")
    parser.add_argument("--patch-size", default=16, type=int, help="每个 patch 的大小")
    parser.add_argument("--depth", default=6, type=int, help="Transformer 的层数")
    parser.add_argument(
        "--mlp-ratio",
        default=4.0,
        type=float,
        help="MLP 中隐藏层通道数相比对 embed_dims 的比例",
    )
    parser.add_argument("--num-heads", default=16, type=int, help="多头注意力数量")
    parser.add_argument("--input-size", default=224, type=int, help="缩放后的图片尺寸")

    args = parser.parse_args()
    return args