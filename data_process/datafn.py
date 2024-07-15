import os
import shutil
from collections import defaultdict
import random
def clear_directory(directory):
    """清空指定目录下的所有文件和子目录"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除目录及其内容
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
# 定义源文件夹和目标文件夹路径
source_dir = '/ima'
target_ff_dir = '/ff'
target_ee_dir = '/ee'
# 清空 ff 和 ee 文件夹的内容
clear_directory(target_ff_dir)
clear_directory(target_ee_dir)
# 获取源文件夹中所有图片文件名
# 获取源文件夹中所有图片文件名
files = os.listdir(source_dir)
image_files = [f for f in files if f.endswith('.JPG')]

# 创建一个字典来存储每个前缀对应的文件列表
prefix_map = defaultdict(list)

# 根据 _A_ 前面的部分进行划分
for filename in image_files:
    prefix = filename.split('_A_')[0]
    prefix_map[prefix].append(filename)

# 遍历前缀字典，按比例划分到 ff 和 ee 文件夹
for prefix, filenames in prefix_map.items():
    random.shuffle(filenames)  # 打乱顺序，保证随机性

    split_idx = int(len(filenames) * 0.8)  # 划分索引
    ff_files = filenames[:split_idx]
    ee_files = filenames[split_idx:]

    # 复制文件到目标文件夹
    for ff_file in ff_files:
        source_path = os.path.join(source_dir, ff_file)
        target_path = os.path.join(target_ff_dir, ff_file)
        shutil.copy(source_path, target_path)
        print(f'Copied {ff_file} to {target_ff_dir}')

    for ee_file in ee_files:
        source_path = os.path.join(source_dir, ee_file)
        target_path = os.path.join(target_ee_dir, ee_file)
        shutil.copy(source_path, target_path)
        print(f'Copied {ee_file} to {target_ee_dir}')