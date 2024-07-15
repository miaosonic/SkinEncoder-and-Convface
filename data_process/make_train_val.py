import os
import random
import shutil

# 定义目录路径
source_dir = ""
train_dir = ""
val_dir = ""

# 创建训练集和验证集目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取所有文件名
files = os.listdir(source_dir)

# 将文件名按照规定的格式进行解析，并根据信息分类
data = {}
for file in files:
    parts = file.split("_")
    age = int(parts[0])
    gender = int(parts[1])
    race = int(parts[2])

    key = (age, gender, race)
    if key not in data:
        data[key] = []
    data[key].append(file)

# 将数据分配到训练集和验证集中
for key, filenames in data.items():
    random.shuffle(filenames)
    split_index = int(0.85 * len(filenames))
    train_files = filenames[:split_index]
    val_files = filenames[split_index:]

    # 将文件复制到训练集目录
    for file in train_files:
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(train_dir, file)
        shutil.copyfile(source_path, dest_path)

    # 将文件复制到验证集目录
    for file in val_files:
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(val_dir, file)
        shutil.copyfile(source_path, dest_path)

print("Data split completed.")
