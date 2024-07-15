import os
import shutil

# 源文件夹路径和目标文件夹路径
source_folder = ''
destination_folder = ''

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 确保只处理文件（跳过目录）
    if os.path.isfile(os.path.join(source_folder, filename)):
        # 按下划线分割文件名
        parts = filename.split('_')

        # 检查文件名是否符合要求
        if len(parts) >= 3:
            race = parts[2]
            if race in {'0', '2'}:
                # 构建源文件和目标文件的完整路径
                source_path = os.path.join(source_folder, filename)
                destination_path = os.path.join(destination_folder, filename)

                # 复制文件到目标文件夹
                shutil.copy(source_path, destination_path)

print("文件复制完成。")
