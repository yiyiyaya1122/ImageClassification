import logging
import os
from datetime import datetime

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import Callback, ModelCheckpoint


def custom_test(model: nn.Cell, dataset: ds.ImageFolderDataset) -> float:
    model.set_train(False)

    total_samples = 0
    correct_predictions = 0

    for data in dataset.create_dict_iterator():
        inputs = data["image"]
        labels = data["label"]

        outputs = model(inputs)

        predicted_classes = ops.argmax(outputs, dim=-1)
        correct_predictions += (
            ops.equal(predicted_classes, labels).astype(ms.float32).sum().asnumpy()
        )
        total_samples += labels.shape[0]
    # 计算准确率
    accuracy = correct_predictions / total_samples
    return accuracy


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, prefix, directory, config=None):
        super(CustomModelCheckpoint, self).__init__(prefix, directory, config)

    def step_end(self, run_context):
        # 在每个 epoch 结束时保存 checkpoint
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num

        # 只在每个 epoch 的最后一步保存 checkpoint
        if step_num % cb_params.batch_num == 0:
            file_name = f"{self._prefix}_e{epoch_num}"
            ckpt_path = os.path.join(self._directory, file_name)
            ms.save_checkpoint(cb_params.train_network, f"{ckpt_path}")
            print(f"Checkpoint saved: {ckpt_path}.ckpt")


class LogCallback(Callback):
    def __init__(
        self,
        net,
        logger,
        eval_dataset,
        total_train_steps,
        log_interval=10,
    ):
        super(LogCallback, self).__init__()
        self.net = net
        self.logger = logger
        self.eval_dataset = eval_dataset
        self.log_interval = log_interval
        self.step_count = 0
        self.current_epoch = 0
        self.total_train_steps = total_train_steps
        self.losses = []

    def step_end(self, run_context):
        # 每隔 log_interval 输出一次训练日志
        self.step_count += 1
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        self.losses.append(loss)
        if self.step_count % self.log_interval == 0:
            epoch_num = cb_params.cur_epoch_num
            avg_loss = sum(self.losses) / len(self.losses)
            self.losses = []
            print(
                f"Epoch #{epoch_num}, Step [{self.step_count}/{self.total_train_steps}], Loss: {avg_loss:.6f}"
            )

    def epoch_end(self, run_context):
        # 在每个 epoch 结束后输出验证日志
        self.step_count = 0
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        # result = self.net.eval(self.eval_dataset)
        # accuracy = result["accuracy"]
        accuracy = custom_test(self.net._network, self.eval_dataset)
        self.logger.info(f"Epoch {epoch_num}, Accuracy on val set: {accuracy*100:.4f}%")


def create_dataset(data_path, batch_size=32, image_size=224, shuffle=False):
    dataset = ds.ImageFolderDataset(data_path, shuffle=shuffle)
    transform = [
        vision.Decode(),
        vision.Resize((image_size, image_size)),
        vision.ToTensor(),
        vision.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False
        ),
    ]
    dataset = dataset.map(operations=transform, input_columns="image")
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


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

    # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
