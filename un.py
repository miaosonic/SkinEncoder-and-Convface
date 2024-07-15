import os
import sys
import json
import pickle
import random
import math
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.utils as nn_utils
import torch.nn as nn
def read_train_data(root: str,task:str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_images_path = []
    train_images_label = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]
    genders = set()  # 创建一个空集合来存储唯一的性别标签

    for filename in os.listdir(root):
        if not any(filename.endswith(ext) for ext in supported):
            continue  # 跳过非图片文件

        parts = os.path.splitext(filename)[0].split('_')
        try:
          #  age = int(parts[0])  # 提取年龄
         #   gender = int(parts[1])  # 提取性别
          #  genders.add(gender)  # 将性别添加到集合中
            ethnicity = int(parts[2])  # 提取种族
        except ValueError:
            continue  # 标签无法解析为整数，跳过

        img_path = os.path.join(root, filename)
        train_images_path.append(img_path)
        if task == 'xx':
           # train_images_label.append(age)
            print("唯一的性别标签:", genders)
        elif task == 'gender':
          #  train_images_label.append(gender)
            print("唯一的性别标签:", genders)
        elif task == 'age':
            train_images_label.append(ethnicity)

    print("{} images for training.".format(len(train_images_path)))
    print("唯一的性别标签数量:", len(genders))
    print("唯一的性别标签:", genders)

    return train_images_path, train_images_label




def read_val_data(root: str,task:str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    val_images_path = []
    val_images_label = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for filename in os.listdir(root):
        if not any(filename.endswith(ext) for ext in supported):
            continue  # 跳过非图片文件

        parts = os.path.splitext(filename)[0].split('_')
        try:
         #   age = int(parts[0])  # 提取年龄
         #   gender = int(parts[1])  # 提取性别
            ethnicity = int(parts[2])  # 提取种族
        except ValueError:
            continue  # 标签无法解析为整数，跳过

        img_path = os.path.join(root, filename)
        val_images_path.append(img_path)
        if task == 'xx':
            val_images_label.append(ethnicity)
        elif task == 'gender':
            val_images_label.append(ethnicity)
        elif task == 'age':
            val_images_label.append(ethnicity)

    print("{} images for training.".format(len(val_images_path)))

    return val_images_path, val_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)

def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


class CombinedLoss(nn.Module):
    def __init__(self, sigma=3, lambda_start=0, lambda_end=1, total_steps=10000):
        super(CombinedLoss, self).__init__()
        self.sigma = sigma
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.total_steps = total_steps
        self.current_step = 0

    def forward(self, predicted, target):
        euclidean_loss = 0.5 * (predicted - target) ** 2
        gaussian_loss = 1 - torch.exp(-((predicted - target) ** 2) / (2 * self.sigma ** 2))

        # Update lambda dynamically based on the current step
        lambda_value = self.lambda_start + (self.lambda_end - self.lambda_start) * (
                    self.current_step / self.total_steps)
        self.current_step += 1

        combined_loss = (1 - lambda_value) * euclidean_loss + lambda_value * gaussian_loss
        return combined_loss.mean()

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, clip_value=None):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        sample_num += images.shape[0]

        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        loss.backward()
        if not torch.isfinite(loss):
            print(f'警告：在步骤 {step} 发现非有限损失，损失值: {loss.item()}')
            # 跳过此批次并继续
            optimizer.zero_grad()
            continue

        # 检查非有限梯度，如果发现则跳过更新
        found_non_finite = False
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print(f'警告：在参数 {name} 中发现非有限梯度')
                found_non_finite = True
                break
        if found_non_finite:
            optimizer.zero_grad()
            continue
        # 梯度裁剪
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        accu_loss += loss.detach()

        data_loader.desc = "[训练 epoch {}] 损失: {:.3f}, 准确率: {:.3f}, 学习率: {:}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
    # 更新学习率
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def read_data(data_path, task):
    images_path = []
    images_label = []

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(('.JPG', '.jpg', '.jpeg')):  # 支持的图像文件格式
                path = os.path.join(root, file)
                # 假设文件名格式为 id_A_Age
                parts = file.split('_')
                if len(parts) == 3:
                    try:
                        id_part = parts[0]
                        age_part = parts[2].split('.')[0]  # 去掉文件扩展名
                        age = int(age_part)
                        images_path.append(path)
                        images_label.append(age)
                    except ValueError:
                        print(f"Skipping file {file}: unable to parse age.")

    return images_path, images_label
def train_one_epoch_age(model, optimizer, data_loader, device, epoch, lr_scheduler, clip_value=None, total_steps=10000):
    model.train()
    loss_function = CombinedLoss(total_steps=total_steps)
    accu_loss = torch.zeros(1).to(device)
    accu_mae = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    # 计算总步数（假设每个epoch的步数相同）
    steps_per_epoch = len(data_loader)
    loss_function.total_steps = total_steps * steps_per_epoch

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device).float()  # 确保标签是浮点数类型

        sample_num += images.shape[0]

        pred = model(images).squeeze()  # 假设模型输出是单个值

        loss = loss_function(pred, labels)
        loss.backward()
        if not torch.isfinite(loss):
            print(f'Warning: Non-finite loss detected at step {step}, loss: {loss.item()}')
            # Skip this batch and continue
            optimizer.zero_grad()
            continue

        # Check for non-finite gradients and skip update if found
        found_non_finite = False
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print(f'Warning: Non-finite gradient detected in parameter {name}')
                found_non_finite = True
                break
        if found_non_finite:
            optimizer.zero_grad()
            continue

        # Gradient clipping
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        accu_loss += loss.detach()
        accu_mae += torch.abs(pred - labels).sum()

        data_loader.desc = "[Train epoch {}] Loss: {:.3f}, MAE: {:.3f}, LR: {:}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_mae.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_mae.item() / sample_num

class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
     #   images, age, labels, ethnicity = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
def evaluate_age(model, data_loader, device, epoch, total_steps=10000):
    loss_function = CombinedLoss(total_steps=total_steps).to(device)
    model.eval()
    accu_loss = torch.zeros(1).to(device)
    accu_mae = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device).float()  # 确保标签是浮点数类型

            sample_num += images.shape[0]

            pred = model(images).squeeze()  # 假设模型输出是单个值

            loss = loss_function(pred, labels)
            accu_loss += loss
            accu_mae += torch.abs(pred - labels).sum()

            data_loader.desc = "[Valid epoch {}] Loss: {:.3f}, MAE: {:.3f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_mae.item() / sample_num
            )

    return accu_loss.item() / (step + 1), accu_mae.item() / sample_num

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        hold_epoch=3,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-2):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        elif (warmup_epochs+hold_epoch)*num_step <= x:
            current_step = (x - (warmup_epochs + hold_epoch) * num_step)
            cosine_steps = (epochs - warmup_epochs - hold_epoch) * num_step  # 减去5轮目标学习率训练的步数
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
        else:
            return 1  # 5轮目标学习率训练后保持不变

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
