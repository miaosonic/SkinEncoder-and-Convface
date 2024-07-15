from sklearn.metrics import r2_score
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # Windows 系统中的黑体字体路径
SEED = 42
data = pd.read_csv('./')
# 找到 Bodyparts 列中值为 'armpit', 'Scalp', 'scalp' 的行索引
#print(data.shape)
indices_to_remove = data[(data['Bodyparts'] == 'armpit') | (data['Bodyparts'] == 'Scalp')| (data['skin_layer'] == 'ET1973') | (data['Bodyparts'] == 'scalp')|(data['Bodyparts'] == 'upper inner arm1990')| (data['Bodyparts'] == 'shoulder')|(data['thickness'].isnull())| (data['Bodyparts'] == 'faceRCM')|(data['Bodyparts'] == 'back of hand')|(data['Bodyparts'] == 'neck')|(data['Bodyparts'] == 'foot')].index

# 删除这些行 data (18313, 6)
# data1 (18190, 6)
data.drop(indices_to_remove, inplace=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import random
# 将'skin_layer'列中的'cheek'转为'face'
data['Bodyparts'] = data['Bodyparts'].replace('cheek', 'face')

# 再次找到'age'列数据为空的行
missing_age_rows = data[data['age'].isnull()]
# 对于每个缺失年龄的行
for index, row in missing_age_rows.iterrows():
    # 找到除了'age'列以外其他列与该行相同的其他行
    matching_rows = data[(data['sex'] == row['sex']) &
                         (data['skin_layer'] == row['skin_layer']) &
                         (data['race'] == row['race']) &
                         (data['Bodyparts'] == row['Bodyparts']) &
                         (~data['age'].isnull())]  # 排除空白值

    if not matching_rows.empty:
        # 计算该行与其他行的“thickness”值的差异
        thickness_difference = abs(matching_rows['thickness'] - row['thickness'])
        # 找到与该行“thickness”值差异最小的行的索引
        min_difference_index = thickness_difference.idxmin()
        # 取出该行的年龄值
        age_value = data.loc[min_difference_index, 'age']
        # 将该年龄值填充到缺失的年龄列中
        data.loc[index, 'age'] = age_value
    if matching_rows.empty:
        matching_rows2 = missing_age_rows[(missing_age_rows['sex'] == row['sex']) &
                             (missing_age_rows['skin_layer'] == row['skin_layer']) &
                             (missing_age_rows['race'] == row['race'])
                            ]
        sorted_group = matching_rows2.sort_values(by='thickness')
        # 计算'thickness'最大值和最小值
        max_thickness = sorted_group['thickness'].max()
        min_thickness = sorted_group['thickness'].min()
        max_thickness_index = sorted_group['thickness'].idxmax()
        # 将最大'thickness'对应的行的年龄设为20
        data.loc[max_thickness_index, 'age'] = 20
        for idx, r in sorted_group.iterrows():
            if idx != max_thickness_index:
                # 计算当前行的'thickness'相对于最大'thickness'的比例
                thickness_ratio = (r['thickness'] - min_thickness) / (max_thickness - min_thickness)
                # 计算年龄值
                age_value = int(80 - thickness_ratio * 60)  # 80 - 20 = 60
                # 设置年龄值
                data.loc[idx, 'age'] = age_value


age_bins = [0, 15, 38, 62, 100]
# 定义对应的标签

age_labels = ['young', 'middle-aged', 'senior', 'elderly']

# 使用 cut 函数对年龄进行分组并添加新列
data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)


missing_sex_rows = data[data['sex'].isnull()]

# 删除'sex'列和'age'列
missing_sex_rows_without_sex_age = missing_sex_rows.drop(columns=['sex', 'age'])
# print(missing_sex_rows_without_sex_age)
# 根据'race', 'Bodyparts', 'skin_layer' 列进行分组
grouped_missing_sex_rows = missing_sex_rows_without_sex_age.groupby(['race', 'Bodyparts', 'skin_layer','age_group'])
# print(grouped_missing_sex_rows)
# 计算每个分组的'thickness'的80%最大值
thresholds = grouped_missing_sex_rows['thickness'].max() * 0.8
for group_key, group_data in grouped_missing_sex_rows:
    # 获取该分组的阈值
    threshold = thresholds[group_key]
    # 遍历该分组的每一行
    for index, row in group_data.iterrows():
        # 如果该行的'thickness'值大于等于70%的阈值，则将'sex'设为'male'
        if row['thickness'] >= 0.8 * threshold:
            data.loc[index, 'sex'] = 'male'
        else:
            # 否则随机选择'sex'为'male'或'female'
            data.loc[index, 'sex'] = random.choice(['male', 'female'])
#clean_df = clean_data(raw_dataframe)
# clean_df.shape # (3030, 27582)
missing_values = data.isnull().sum()
print('mi',missing_values)
grouped_data = data.groupby(['sex', 'skin_layer', 'race', 'Bodyparts','age_group'])

# 对于每个分组，检查 'thickness' 最大值是否大于最小值的两倍
for group, group_data in grouped_data:
    max_thickness = group_data['thickness'].max()
    min_thickness = group_data['thickness'].min()
    if max_thickness > 4 * min_thickness:
        # 计算四分之一到四分之三之间的值
        range_min = min_thickness + (max_thickness - min_thickness) * 0.4
        range_max = min_thickness + (max_thickness - min_thickness) * 0.6
        # 将所有值映射到四分之一到四分之三之间
        group_data['thickness'] = np.clip(group_data['thickness'], range_min, range_max)
        # 更新原始数据中的值
        data.loc[group_data.index, 'thickness'] = group_data['thickness']
        '''
            if max_thickness > 2 * min_thickness:
        # 计算四分之一到四分之三之间的值
        range_min = min_thickness + (max_thickness - min_thickness) * 0.25
        range_max = min_thickness + (max_thickness - min_thickness) * 0.75
        # 将所有值映射到四分之一到四分之三之间
        group_data['thickness'] = np.clip(group_data['thickness'], range_min, range_max)
        # 更新原始数据中的值
        data.loc[group_data.index, 'thickness'] = group_data['thickness']
        '''


# 对分类变量进行独热编码

import matplotlib.pyplot as plt


data = pd.get_dummies(data, columns=['Bodyparts', 'skin_layer','age_group','sex','race'])
data = data.rename(columns={'race_eu': 'thickness', 'thickness': 'race_eu'})
data['race_eu'], data['thickness'] = data['thickness'], data['race_eu']


# 筛选所有前缀为 'Bodyparts_' 的列
bodyparts_columns = [col for col in data.columns if col.startswith('Bodyparts_')]
skin_layer_columns = [col for col in data.columns if col.startswith('skin_layer_')]

# 创建一个字典来存储每个 Bodyparts 列在每个 skin_layer 下的 'thickness' 范围
thickness_ranges = {}

# 计算每个 Bodyparts 列在每个 skin_layer 下为 1 时的 'thickness' 范围
for bodyparts_col in bodyparts_columns:
    for skin_layer_col in skin_layer_columns:
        selected_data = data[(data[bodyparts_col] == 1) & (data[skin_layer_col] == 1) & (data['race_eu'] == 1)]
        if not selected_data.empty:
            min_thickness = selected_data['thickness'].min()
            max_thickness = selected_data['thickness'].max()
            thickness_ranges[(bodyparts_col, skin_layer_col)] = (min_thickness, max_thickness)
            print(f"{bodyparts_col} 和 {skin_layer_col} 为 1 时的 'thickness' 范围: {min_thickness} - {max_thickness}")
            plt.figure(figsize=(10, 6))
            plt.scatter(selected_data['age'], selected_data['thickness'], marker='o', color='b', alpha=0.5)
            plt.title(f"{bodyparts_col} 和 {skin_layer_col} 为 1 时的 'thickness' 随 'age' 的变化")
            plt.xlabel('Age')
            plt.ylabel('Thickness')
            plt.grid(True)
            plt.show()



# 排除 'thickness' 和 'age' 列，其他列用于分组
group_columns = [col for col in data.columns if col not in ['thickness', 'age']]

# 分组
grouped_data = data.groupby(group_columns)

train_list = []
val_list = []

# 设置随机种子以保证结果可复现
np.random.seed(SEED)

# 按组进行拆分
for name, group in grouped_data:
    # 按90%和10%的比例拆分
    train_size = int(0.9 * len(group))

    # 获取每个组的索引
    indices = np.arange(len(group))

    # 打乱索引，但不改变数据顺序
    np.random.shuffle(indices)

    # 获取训练集和验证集的索引
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 根据索引选择训练集和验证集
    train_group = group.iloc[train_indices]
    val_group = group.iloc[val_indices]

    train_list.append(train_group)
    val_list.append(val_group)

# 合并所有分组的训练集和验证集
train = pd.concat(train_list)
validate = pd.concat(val_list)

# 确保索引连续
train.reset_index(drop=True, inplace=True)
validate.reset_index(drop=True, inplace=True)

# 转换年龄列为float64类型
train_ages = train['age'].astype('float64')
val_ages = validate['age'].astype('float64')

class BaseDataset(Dataset):
    df = None
    ages = None
    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int):
        return torch.tensor(self.df_np.iloc[idx, :-1].values.astype('float32')), torch.tensor(self.ages_np.iloc[idx], dtype=torch.float32)

class TrainDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.df = torch.from_numpy(np.array(train).astype('float32'))
        self.df_np = train
        self.ages_np = train_ages
        self.ages = torch.from_numpy(np.array(train_ages).astype('float32'))

class ValDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.df = torch.from_numpy(np.array(validate).astype('float32'))
        self.df_np = validate
        self.ages_np = val_ages
        self.ages = torch.from_numpy(np.array(val_ages).astype('float32'))
'''
class TestDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.df = torch.from_numpy(np.array(test).astype('float32'))
        self.df_np = test
        self.ages_np = test_ages
        self.ages = torch.from_numpy(np.array(test_ages).astype('float32'))
'''


# Configuration dictionary
cfg = edict({
    'input_dim': 24,
    'nhid': 24,
    'nlayers':4 ,
    'nhead': 4,
    'attn_dropout': 0,
    'resid_dropout': 0,
    'embd_dropout': 0,
    'nff': 24,
    'lr': 0.005,
    'gpu': True,
    'batch_size': 128,
    'beta1': 0.9,
    'beta2': 0.98,
    'eps': 1e-9
})

class MultiheadAttention(nn.Module):
    def __init__(self, mask=False):
        super(MultiheadAttention, self).__init__()
        assert cfg.nhid % cfg.nhead == 0
        self.nhead = cfg.nhead
        self.head_dim = cfg.nhid // cfg.nhead
        self.key = nn.Linear(cfg.nhid, cfg.nhid)
        self.query = nn.Linear(cfg.nhid, cfg.nhid)
        self.value = nn.Linear(cfg.nhid, cfg.nhid)
        self.dropout = nn.Dropout(cfg.attn_dropout)
        self.ln1 = nn.LayerNorm(cfg.nhid)

    def forward(self, q, k, v, mask=None):
        c = q
    #    print('c',c.shape)
        q = self.query(q)
        bs, q_len = q.size()
        q = torch.reshape(q, (bs, cfg.nhead, self.head_dim, 1))
        k = self.key(k)
        k = torch.reshape(k, (bs, cfg.nhead, self.head_dim, 1))
        v = self.value(v)
        v = torch.reshape(v, (bs, cfg.nhead, self.head_dim, 1))
        mat_mul_1 = torch.matmul(q, k.transpose(-2, -1))
        scale = mat_mul_1 / (self.head_dim ** 0.5)

        if mask is not None:
            scale = scale.masked_fill(mask == 0, float("-inf"))

        softm = torch.relu(scale)
      #  softm=torch.softmax(scale, dim=1)
      #  scale = scale.reshape(bs, self.nhead * self.head_dim)
        dropout = self.dropout(softm)
        mat_mul_2 = torch.matmul(dropout, v)
        tr = torch.transpose(mat_mul_2, 1, 2)
        concat_attn = tr.reshape(bs, self.nhead * self.head_dim)
        outputs = concat_attn + c
        outputs = self.ln1(outputs)

        return outputs

class Feedforward(nn.Module):
    def __init__(self):
        super(Feedforward, self).__init__()
        self.ln1 = nn.Linear(cfg.nhid, 2 * cfg.nff)
        self.relu = nn.ReLU()
        self.ln2 = nn.Linear(2 * cfg.nff, cfg.nhid)
        self.dropout = nn.Dropout(cfg.resid_dropout)

    def forward(self, x):
        return self.dropout(self.ln2(self.relu(self.ln1(x))))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.ln1 = nn.LayerNorm(cfg.nhid)
        self.ln2 = nn.LayerNorm(cfg.nhid)
        self.self_attn = MultiheadAttention()
        self.dropout = nn.Dropout(cfg.resid_dropout)
        self.ff = Feedforward()

    def forward(self, x, mask=None):
        ln1_out = self.ln1(x)
        q = ln1_out.clone()
        k = ln1_out.clone()
        v = ln1_out.clone()
        masked_mha = self.self_attn(q, k, v, mask)
        dropout = self.dropout(masked_mha)
        output = dropout + ln1_out
        ln2_out = self.ln2(output)
        ff = self.ff(ln2_out)
        outputs = self.ln2(ff + ln2_out)
        return outputs

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(cfg.embd_dropout)
        self.transform = nn.ModuleList([EncoderLayer() for _ in range(cfg.nlayers)])
        self.ln_f = nn.LayerNorm(cfg.nhid)
        self.output_layer = nn.Linear(cfg.nhid, 1)
        self.ln3 = nn.Linear(cfg.input_dim, cfg.nhid)

    def forward(self, x, mask=None):
        x = self.ln3(x)
        drop = self.dropout(x)
        nlayers = len(self.transform)
        out = drop
        for i in range(nlayers):
            out = self.transform[i](out, mask)
        out = self.ln_f(out)
        out = self.output_layer(out)
        out = torch.squeeze(out, dim=-1)
        return out

# Initialize DataLoader instances
train_loader = DataLoader(TrainDataset(), batch_size=cfg.batch_size)
val_loader = DataLoader(ValDataset(), batch_size=cfg.batch_size)
#test_loader = DataLoader(TestDataset(), batch_size=cfg.batch_size)

# Instantiate model and move it to GPU if available
model = Encoder()
device = torch.device("cuda" if torch.cuda.is_available() and cfg.gpu else "cpu")
model.to(device)
if device.type == 'cuda':
    print("Using GPU for training!")
else:
    print("Using CPU for training!")
# Define optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (x, label) in enumerate(train_loader):
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        prediction = model(x)
        loss = nn.MSELoss()(prediction, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
     #   if batch_idx % 10 == 0:
         #   print(f"Batch {batch_idx}, Loss: {loss.item()}")
    return running_loss / len(train_loader)
from sklearn.metrics import mean_absolute_error
def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_r2 = 0.0
    val_mae = 0.0
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(val_loader):
            x, label = x.to(device), label.to(device)
            prediction = model(x)
            loss = nn.MSELoss()(prediction, label)
            val_loss += loss.item()
            val_r2 += r2_score(label.cpu().numpy(), prediction.cpu().numpy())
            val_mae += mean_absolute_error(label.cpu().numpy(), prediction.cpu().numpy())
    return val_loss / len(val_loader), val_r2 / len(val_loader), val_mae / len(val_loader)

num_epochs = 20
best_val_r2 = float('-inf')
best_val_mae = float('inf')  # 初始化为正无穷大，以找到最低的 MAE
best_model_path = "best_model.pth"

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_r2, val_mae = validate(model, val_loader, device)
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    if val_mae < best_val_mae:  # 如果当前 MAE 比记录的最低 MAE 还低，更新最低 MAE 和保存模型
        best_val_mae = val_mae
        best_val_r2 = val_r2
        torch.save(model.state_dict(), best_model_path)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val R2: {val_r2}, Val MAE: {val_mae}, Learning Rate: {current_lr}")

print(f"Best Val R2: {best_val_r2}, Best Val MAE: {best_val_mae}")

'''
# Test Loop with R2 score
model.eval()
errors = []
all_labels = []
all_predictions = []
with torch.no_grad():
    for x, b in test_loader:
        x, b = x.to(device), b.to(device)
        pred = model(x)
        diff = torch.abs(pred - b)
        errors.extend(diff.cpu().numpy())
        all_labels.extend(b.cpu().numpy())
        all_predictions.extend(pred.cpu().numpy())

errors = np.array(errors)
mean_error = np.mean(errors)
median_error = np.median(errors)

# Calculate R2 score
r2 = r2_score(np.array(all_labels), np.array(all_predictions))

print(f"Mean Error: {mean_error}")
print(f"Median Error: {median_error}")
print(f"R2 Score: {r2}")
'''
