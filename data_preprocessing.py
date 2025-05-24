import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import os
from tqdm import tqdm
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from cycler import cycler
import matplotlib.font_manager as fm
import matplotlib
import sys
import logging
from datetime import datetime

# 设置精美的图表样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#DDDDDD'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = '#CCCCCC'

# 设置精美的颜色循环
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#d35400', '#c0392b', '#7f8c8d']
plt.rcParams['axes.prop_cycle'] = cycler('color', colors)

# 创建结果文件夹
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/figures'):
    os.makedirs('results/figures')
if not os.path.exists('results/figures/user_analysis'):
    os.makedirs('results/figures/user_analysis')
if not os.path.exists('results/figures/item_analysis'):
    os.makedirs('results/figures/item_analysis')
if not os.path.exists('results/figures/time_analysis'):
    os.makedirs('results/figures/time_analysis')
if not os.path.exists('results/figures/network_analysis'):
    os.makedirs('results/figures/network_analysis')
if not os.path.exists('results/figures/limited_range'):
    os.makedirs('results/figures/limited_range')

# 检查文件是否存在以及文件格式
def check_file(file_path):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    try:
        # 读取前5行检查格式
        with open(file_path, 'r', encoding='utf-8') as f:
            for i in range(5):
                line = f.readline().strip()
                if not line:
                    continue
                json_obj = json.loads(line)
                print(f"文件 {file_path} 第{i+1}行示例: {json_obj}")
        return True
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return False

# 检查数据文件
print("检查训练数据文件...")
train_file = 'data/train/item_share_train_info.json'
user_file = 'data/train/user_info.json'
item_file = 'data/train/item_info.json'
test_file = 'data/test/item_share_preliminary_test_info.json'

if not check_file(train_file) or not check_file(user_file) or not check_file(item_file) or not check_file(test_file):
    print("数据文件检查失败，请确保文件存在且格式正确")
    
# 加载数据
print("加载训练数据...")

# 分批读取较大的JSON文件
def load_json_in_chunks(file_path, chunk_size=100000):
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        for i, line in tqdm(enumerate(f)):
            try:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                # 确保每行都被解析为JSON对象
                json_obj = json.loads(line)
                # 如果是列表，展开它
                if isinstance(json_obj, list):
                    chunk.extend(json_obj)  # 扩展列表中的所有项
                else:
                    chunk.append(json_obj)  # 添加单个对象
                    
                if (i + 1) % chunk_size == 0 and chunk:
                    chunks.append(pd.DataFrame(chunk))
                    chunk = []
            except Exception as e:
                print(f"Error parsing line {i}: {e}, Content: {line[:100]}")
        
        if chunk:  # 添加最后一个不完整的chunk
            chunks.append(pd.DataFrame(chunk))
    
    if not chunks:
        print(f"警告: 从 {file_path} 读取的数据为空!")
        return pd.DataFrame()
    
    result = pd.concat(chunks, ignore_index=True)
    print(f"从 {file_path} 成功读取 {len(result)} 行数据")
    return result

# 从文件读取数据
def read_json_to_df(file_path, chunk_size=None):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return pd.DataFrame()
    
    print(f"正在从 {file_path} 读取数据...")    
    if chunk_size:
        return load_json_in_chunks(file_path, chunk_size)
    else:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f)):
                try:
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                        
                    # 确保每行都被解析为JSON对象
                    json_obj = json.loads(line)
                    # 如果是列表，展开它
                    if isinstance(json_obj, list):
                        data.extend(json_obj)  # 扩展列表中的所有项
                    else:
                        data.append(json_obj)  # 添加单个对象
                except Exception as e:
                    print(f"Error parsing line {i}: {e}, Content: {line[:100]}")
        
        if not data:
            print(f"警告: 从 {file_path} 读取的数据为空!")
            return pd.DataFrame()
            
        result = pd.DataFrame(data)
        print(f"从 {file_path} 成功读取 {len(result)} 行数据")
        return result

# 加载训练集数据
train_data = read_json_to_df('data/train/item_share_train_info.json', chunk_size=200000)
user_info = read_json_to_df('data/train/user_info.json')
item_info = read_json_to_df('data/train/item_info.json', chunk_size=100000)

# 加载初赛测试集数据（忽略复赛数据集）
test_data = read_json_to_df('data/test/item_share_preliminary_test_info.json')

# 查看训练数据的列名
print("训练数据列名:", train_data.columns.tolist())

# 数据预处理
print("数据预处理...")

# 如果列名是数字，将列重命名为实际的字段名
if all(isinstance(col, int) for col in train_data.columns) or all(isinstance(col, str) and col.isdigit() for col in train_data.columns):
    print("检测到训练数据列名为数字，重命名列...")
    # 根据第一行数据获取原始的列名
    if len(train_data) > 0:
        try:
            with open('data/train/item_share_train_info.json', 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                first_json = json.loads(first_line)
                # 检查是字典还是列表
                if isinstance(first_json, dict):
                    original_keys = list(first_json.keys())
                else:
                    # 如果是列表，使用DataFrame中数据的列名
                    original_keys = train_data.iloc[0].index.tolist()
            
            column_mapping = {i: key for i, key in enumerate(original_keys)}
            train_data = train_data.rename(columns=column_mapping)
        except Exception as e:
            print(f"从文件获取列名失败: {e}")
            # 直接设置列名为已知的列名
            known_columns = ['inviter_id', 'item_id', 'voter_id', 'timestamp']
            if len(train_data.columns) == len(known_columns):
                train_data.columns = known_columns
                print("已使用预定义的列名")
            else:
                print(f"无法自动识别列名。当前列数: {len(train_data.columns)}, 预定义列数: {len(known_columns)}")
        
        print("重命名后的列名:", train_data.columns.tolist())

# 处理user_info的列名
if all(isinstance(col, int) for col in user_info.columns) or all(isinstance(col, str) and col.isdigit() for col in user_info.columns):
    print("检测到用户信息列名为数字，重命名列...")
    if len(user_info) > 0:
        try:
            with open('data/train/user_info.json', 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                first_json = json.loads(first_line)
                # 检查是字典还是列表
                if isinstance(first_json, dict):
                    original_keys = list(first_json.keys())
                else:
                    # 如果是列表，使用DataFrame中数据的列名
                    original_keys = user_info.iloc[0].index.tolist()
            
            column_mapping = {i: key for i, key in enumerate(original_keys)}
            user_info = user_info.rename(columns=column_mapping)
        except Exception as e:
            print(f"从文件获取列名失败: {e}")
            # 直接设置列名为已知的列名
            known_columns = ['user_id', 'user_gender', 'user_age', 'user_level']
            if len(user_info.columns) == len(known_columns):
                user_info.columns = known_columns
                print("已使用预定义的列名")
            else:
                print(f"无法自动识别列名。当前列数: {len(user_info.columns)}, 预定义列数: {len(known_columns)}")
        
        print("重命名后的用户信息列名:", user_info.columns.tolist())

# 处理item_info的列名
if all(isinstance(col, int) for col in item_info.columns) or all(isinstance(col, str) and col.isdigit() for col in item_info.columns):
    print("检测到商品信息列名为数字，重命名列...")
    if len(item_info) > 0:
        try:
            with open('data/train/item_info.json', 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                first_json = json.loads(first_line)
                # 检查是字典还是列表
                if isinstance(first_json, dict):
                    original_keys = list(first_json.keys())
                else:
                    # 如果是列表，使用DataFrame中数据的列名
                    original_keys = item_info.iloc[0].index.tolist()
            
            column_mapping = {i: key for i, key in enumerate(original_keys)}
            item_info = item_info.rename(columns=column_mapping)
        except Exception as e:
            print(f"从文件获取列名失败: {e}")
            # 直接设置列名为已知的列名
            known_columns = ['item_id', 'cate_id', 'cate_level1_id', 'brand_id']
            if len(item_info.columns) == len(known_columns):
                item_info.columns = known_columns
                print("已使用预定义的列名")
            else:
                print(f"无法自动识别列名。当前列数: {len(item_info.columns)}, 预定义列数: {len(known_columns)}")
        
        print("重命名后的商品信息列名:", item_info.columns.tolist())

# 处理test_data的列名
if all(isinstance(col, int) for col in test_data.columns) or all(isinstance(col, str) and col.isdigit() for col in test_data.columns):
    print("检测到测试数据列名为数字，重命名列...")
    if len(test_data) > 0:
        try:
            with open('data/test/item_share_preliminary_test_info.json', 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                first_json = json.loads(first_line)
                # 检查是字典还是列表
                if isinstance(first_json, dict):
                    original_keys = list(first_json.keys())
                else:
                    # 如果是列表，使用DataFrame中数据的列名
                    original_keys = test_data.iloc[0].index.tolist()
            
            column_mapping = {i: key for i, key in enumerate(original_keys)}
            test_data = test_data.rename(columns=column_mapping)
        except Exception as e:
            print(f"从文件获取列名失败: {e}")
            # 直接设置列名为已知的列名
            known_columns = ['inviter_id', 'item_id', 'voter_id', 'timestamp']
            if len(test_data.columns) == len(known_columns):
                test_data.columns = known_columns
                print("已使用预定义的列名")
            else:
                print(f"无法自动识别列名。当前列数: {len(test_data.columns)}, 预定义列数: {len(known_columns)}")
        
        print("重命名后的测试数据列名:", test_data.columns.tolist())

# 转换时间戳为datetime类型
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])

# 提取日期特征
train_data['date'] = train_data['timestamp'].dt.date
train_data['hour'] = train_data['timestamp'].dt.hour
train_data['day_of_week'] = train_data['timestamp'].dt.dayofweek

# 合并用户和商品信息到训练数据
train_with_user = pd.merge(train_data, user_info, left_on='inviter_id', right_on='user_id', how='left')
train_full = pd.merge(train_with_user, item_info, left_on='item_id', right_on='item_id', how='left')

# 数据可视化
print("生成数据可视化...")

# 1. 用户性别分布
plt.figure(figsize=(10, 6))
gender_counts = user_info['user_gender'].value_counts()
gender_labels = {-1: '未知', 0: '女性', 1: '男性'}
gender_counts.index = [gender_labels[i] for i in gender_counts.index]

ax = sns.barplot(x=gender_counts.index, y=gender_counts.values, palette=['#95a5a6', '#e74c3c', '#3498db'])
plt.title('用户性别分布', fontsize=16, fontweight='bold')
plt.ylabel('用户数量', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数据标签
for i, v in enumerate(gender_counts.values):
    ax.text(i, v + 0.1, f'{v:,}', ha='center', fontsize=10)

# 添加百分比
total = gender_counts.sum()
for i, v in enumerate(gender_counts.values):
    percentage = v / total * 100
    ax.text(i, v/2, f'{percentage:.1f}%', ha='center', color='white', fontweight='bold')

# 手动设置x轴刻度标签字体
plt.xticks(plt.xticks()[0], gender_counts.index)

plt.savefig('results/figures/user_gender_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 用户年龄段分布

plt.figure(figsize=(12, 6))
age_counts = user_info['user_age'].value_counts().sort_index()
age_counts.index = ['未知' if i == -1 else f'年龄段{i}' for i in age_counts.index]
ax = sns.barplot(x=age_counts.index, y=age_counts.values, palette=sns.color_palette("viridis", len(age_counts)))
plt.title('用户年龄段分布', fontsize=16, fontweight='bold')
plt.ylabel('用户数量', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数据标签
for i, v in enumerate(age_counts.values):
    ax.text(i, v + 0.1, f'{v:,}', ha='center', fontsize=9, rotation=45)

plt.savefig('results/figures/user_age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 用户等级分布
plt.figure(figsize=(12, 6))
level_counts = user_info['user_level'].value_counts().sort_index()
ax = sns.barplot(x=level_counts.index, y=level_counts.values, palette=sns.color_palette("YlOrRd", len(level_counts)))
plt.title('用户等级分布', fontsize=16, fontweight='bold')
plt.xlabel('用户等级', fontsize=12)
plt.ylabel('用户数量', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数据标签
for i, v in enumerate(level_counts.values):
    ax.text(i, v + 0.1, f'{v:,}', ha='center', fontsize=9)

plt.savefig('results/figures/user_level_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 商品类别分布（取前20个）
plt.figure(figsize=(14, 7))
category_counts = item_info['cate_id'].value_counts().nlargest(20)
ax = sns.barplot(x=category_counts.index, y=category_counts.values, palette=sns.color_palette("Blues_r", 20))
plt.title('商品类别分布（前20个）', fontsize=16, fontweight='bold')
plt.xlabel('类别ID', fontsize=12)
plt.ylabel('商品数量', fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数据标签
for i, v in enumerate(category_counts.values):
    ax.text(i, v/2, f'{v:,}', ha='center', color='white', fontsize=8, fontweight='bold')

plt.savefig('results/figures/item_category_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 一级类目分布（取前20个）
plt.figure(figsize=(14, 7))
category1_counts = item_info['cate_level1_id'].value_counts().nlargest(20)
ax = sns.barplot(x=category1_counts.index, y=category1_counts.values, palette=sns.color_palette("Greens_r", 20))
plt.title('一级类目分布（前20个）', fontsize=16, fontweight='bold')
plt.xlabel('一级类目ID', fontsize=12)
plt.ylabel('商品数量', fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数据标签
for i, v in enumerate(category1_counts.values):
    ax.text(i, v/2, f'{v:,}', ha='center', color='white', fontsize=8, fontweight='bold')

plt.savefig('results/figures/item_category1_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. 品牌分布（取前20个）
plt.figure(figsize=(14, 7))
brand_counts = item_info['brand_id'].value_counts().nlargest(20)
ax = sns.barplot(x=brand_counts.index, y=brand_counts.values, palette=sns.color_palette("Purples_r", 20))
plt.title('品牌分布（前20个）', fontsize=16, fontweight='bold')
plt.xlabel('品牌ID', fontsize=12)
plt.ylabel('商品数量', fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数据标签
for i, v in enumerate(brand_counts.values):
    ax.text(i, v/2, f'{v:,}', ha='center', color='white', fontsize=8, fontweight='bold')

plt.savefig('results/figures/item_brand_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. 时间分布分析
# 按日期统计分享数量
try:
    print("生成每日分享数量变化图...")

    plt.figure(figsize=(14, 7))
    date_counts = train_data['date'].value_counts().sort_index()
    
    # 创建渐变色
    gradient_line = np.linspace(0, 1, len(date_counts))
    cmap = plt.cm.cool
    
    # 绘制带渐变色的线和点
    for i in range(len(date_counts)-1):
        plt.plot(date_counts.index[i:i+2], date_counts.values[i:i+2], 
                 color=cmap(gradient_line[i]), linewidth=2.5)
    
    # 添加点
    plt.scatter(date_counts.index, date_counts.values, s=30, 
                c=gradient_line, cmap=cmap, zorder=3, edgecolor='white')
    
    plt.title('每日分享数量变化', fontsize=16, fontweight='bold')
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('分享数量', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加平均线
    avg = date_counts.mean()
    plt.axhline(y=avg, color='r', linestyle='--', alpha=0.7)
    plt.text(date_counts.index[0], avg*1.05, f'平均: {avg:.0f}', color='r')
    
    plt.tight_layout()
    plt.savefig('results/figures/sharing_by_date.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成每日分享数量变化图时出错: {e}")

# 按小时统计分享数量
try:
    print("生成各小时分享数量分布图...")

    plt.figure(figsize=(12, 6))
    hour_counts = train_data['hour'].value_counts().sort_index()
    
    # 使用渐变色调色板
    palette = sns.color_palette("plasma", 24)
    ax = sns.barplot(x=hour_counts.index, y=hour_counts.values, palette=palette)
    
    plt.title('各小时分享数量分布', fontsize=16, fontweight='bold')
    plt.xlabel('小时', fontsize=12)
    plt.ylabel('分享数量', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数据标签
    for i, v in enumerate(hour_counts.values):
        ax.text(i, v + 0.1, f'{v:,}', ha='center', fontsize=8, rotation=45)
    
    # 标记高峰时段
    peak_hour = hour_counts.idxmax()
    peak_value = hour_counts.max()
    plt.annotate(f'峰值: {peak_hour}时 ({peak_value:,})',
                xy=(peak_hour, peak_value),
                xytext=(peak_hour-2, peak_value*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10, fontweight='bold')
    
    plt.savefig('results/figures/sharing_by_hour.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成各小时分享数量分布图时出错: {e}")

# 按星期几统计分享数量
try:
    print("生成各星期几分享数量分布图...")

    plt.figure(figsize=(10, 6))
    weekday_counts = train_data['day_of_week'].value_counts().sort_index()
    
    # 确保索引包含所有星期几（0-6）
    for i in range(7):
        if i not in weekday_counts.index:
            weekday_counts[i] = 0
    weekday_counts = weekday_counts.sort_index()
    
    # 创建映射
    weekday_mapping = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
    # 使用映射重命名索引
    weekday_counts.index = [weekday_mapping.get(i, f'未知{i}') for i in weekday_counts.index]
    
    # 使用不同颜色突出周末
    colors = ['#3498db', '#3498db', '#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c']
    ax = sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette=colors)
    
    plt.title('各星期几分享数量分布', fontsize=16, fontweight='bold')
    plt.ylabel('分享数量', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数据标签
    for i, v in enumerate(weekday_counts.values):
        ax.text(i, v + 0.1, f'{v:,}', ha='center', fontsize=9)
    
    # 添加周末与工作日对比注释
    workday_avg = weekday_counts.iloc[:5].mean()
    weekend_avg = weekday_counts.iloc[5:].mean()
    diff_pct = (weekend_avg - workday_avg) / workday_avg * 100
    
    plt.figtext(0.5, 0.01, f'周末平均: {weekend_avg:.0f} | 工作日平均: {workday_avg:.0f} | 差异: {diff_pct:.1f}%', 
                ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 为底部注释留出空间
    plt.savefig('results/figures/sharing_by_weekday.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成各星期几分享数量分布图时出错: {e}")

# 8. 用户活跃度分析
try:
    print("生成用户活跃度分布图...")

    inviter_counts = train_data['inviter_id'].value_counts()
    voter_counts = train_data['voter_id'].value_counts()

    plt.figure(figsize=(16, 8))
    
    # 使用子图1：邀请者活跃度
    plt.subplot(1, 2, 1)
    # 使用更好的颜色和样式
    sns.histplot(inviter_counts, bins=50, kde=True, color='#3498db')
    
    plt.title('邀请者活跃度分布', fontsize=16, fontweight='bold')
    plt.xlabel('邀请次数', fontsize=12)
    plt.ylabel('用户数量', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加统计信息
    avg_invites = inviter_counts.mean()
    median_invites = inviter_counts.median()
    max_invites = inviter_counts.max()
    
    stats_text = f"平均: {avg_invites:.1f}\n中位数: {median_invites:.1f}\n最大值: {max_invites:.0f}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # 使用子图2：回流者活跃度
    plt.subplot(1, 2, 2)
    sns.histplot(voter_counts, bins=50, kde=True, color='#2ecc71')
    
    plt.title('回流者活跃度分布', fontsize=16, fontweight='bold')
    plt.xlabel('回流次数', fontsize=12)
    plt.ylabel('用户数量', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加统计信息
    avg_votes = voter_counts.mean()
    median_votes = voter_counts.median()
    max_votes = voter_counts.max()
    
    stats_text = f"平均: {avg_votes:.1f}\n中位数: {median_votes:.1f}\n最大值: {max_votes:.0f}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/figures/user_activity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 添加对数尺度图表
    plt.figure(figsize=(16, 8))
    
    # 使用子图1：邀请者活跃度（对数尺度）
    plt.subplot(1, 2, 1)
    sns.histplot(np.log1p(inviter_counts), bins=50, kde=True, color='#3498db')
    
    plt.title('邀请者活跃度分布（对数尺度）', fontsize=16, fontweight='bold')
    plt.xlabel('邀请次数（对数）', fontsize=12)
    plt.ylabel('用户数量', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 使用子图2：回流者活跃度（对数尺度）
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(voter_counts), bins=50, kde=True, color='#2ecc71')
    
    plt.title('回流者活跃度分布（对数尺度）', fontsize=16, fontweight='bold')
    plt.xlabel('回流次数（对数）', fontsize=12)
    plt.ylabel('用户数量', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/figures/user_activity_log_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成用户活跃度分布图时出错: {e}")

# 9. 社交网络分析
try:
    print("进行社交网络分析...")
    # 构建图
    G = nx.DiGraph()
    
    # 添加边（限制样本数量以防止内存溢出）
    edge_sample = train_data[['inviter_id', 'voter_id']].sample(n=min(50000, len(train_data)))
    edges = list(zip(edge_sample['inviter_id'], edge_sample['voter_id']))
    G.add_edges_from(edges)
    
    # 计算节点度数
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    

    plt.figure(figsize=(16, 8))
    
    # 入度分布
    plt.subplot(1, 2, 1)
    in_degree_values = list(in_degrees.values())
    
    # 使用更美观的颜色和样式
    sns.histplot(in_degree_values, bins=30, kde=True, color='#3498db')
    
    plt.title('入度分布', fontsize=16, fontweight='bold')
    plt.xlabel('入度', fontsize=12)
    plt.ylabel('节点数量', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加统计信息
    avg_in = np.mean(in_degree_values)
    median_in = np.median(in_degree_values)
    max_in = max(in_degree_values) if in_degree_values else 0
    
    stats_text = f"平均入度: {avg_in:.2f}\n中位数: {median_in:.1f}\n最大值: {max_in}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # 出度分布
    plt.subplot(1, 2, 2)
    out_degree_values = list(out_degrees.values())
    
    sns.histplot(out_degree_values, bins=30, kde=True, color='#2ecc71')
    
    plt.title('出度分布', fontsize=16, fontweight='bold')
    plt.xlabel('出度', fontsize=12)
    plt.ylabel('节点数量', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加统计信息
    avg_out = np.mean(out_degree_values)
    median_out = np.median(out_degree_values)
    max_out = max(out_degree_values) if out_degree_values else 0
    
    stats_text = f"平均出度: {avg_out:.2f}\n中位数: {median_out:.1f}\n最大值: {max_out}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/figures/degree_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 添加对数尺度的度数分布图
    plt.figure(figsize=(16, 8))
    
    # 入度对数分布
    plt.subplot(1, 2, 1)
    # 避免log(0)
    log_in_degrees = [np.log1p(d) for d in in_degree_values if d > 0]
    
    sns.histplot(log_in_degrees, bins=30, kde=True, color='#3498db')
    
    plt.title('入度分布（对数尺度）', fontsize=16, fontweight='bold')
    plt.xlabel('入度（对数）', fontsize=12)
    plt.ylabel('节点数量', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 出度对数分布
    plt.subplot(1, 2, 2)
    # 避免log(0)
    log_out_degrees = [np.log1p(d) for d in out_degree_values if d > 0]
    
    sns.histplot(log_out_degrees, bins=30, kde=True, color='#2ecc71')
    
    plt.title('出度分布（对数尺度）', fontsize=16, fontweight='bold')
    plt.xlabel('出度（对数）', fontsize=12)
    plt.ylabel('节点数量', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/figures/degree_log_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"进行社交网络分析时出错: {e}")

# 添加热力图和趋势图
# 按小时和星期几分布的热力图
if 'hour' in train_data.columns and 'day_of_week' in train_data.columns:
    try:
        print("生成按小时和星期几分布的热力图...")
        # 创建小时和星期几的交叉表
        hour_weekday = pd.crosstab(train_data['hour'], train_data['day_of_week'])
        
        # 确保所有星期几都存在
        for i in range(7):
            if i not in hour_weekday.columns:
                hour_weekday[i] = 0
        
        # 排序列
        hour_weekday = hour_weekday.reindex(sorted(hour_weekday.columns), axis=1)
        
        # 创建星期几名称映射
        weekday_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
        hour_weekday.columns = [weekday_names[col] for col in hour_weekday.columns]
        

        plt.figure(figsize=(12, 10))
        
        # 使用更好的颜色映射和注释格式
        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        ax = sns.heatmap(hour_weekday, cmap=cmap, annot=True, fmt=',d', 
                     linewidths=0.5, cbar_kws={'label': '分享数量'})
        
        # 设置标题和标签
        plt.title('按小时和星期几分布的分享数量热力图', fontsize=16, fontweight='bold')
        plt.xlabel('星期几', fontsize=12)
        plt.ylabel('小时', fontsize=12)
        
        # 找出热点并标记
        max_val = hour_weekday.max().max()
        max_hour = hour_weekday.max(axis=1).idxmax()
        max_day = hour_weekday.max().idxmax()
        
        # 添加热点标记
        plt.text(0.5, -0.1, f'热点时段: {max_hour}时 {max_day} (分享数量: {max_val:,})', 
                 ha='center', transform=ax.transAxes, fontsize=12, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('results/figures/time_analysis/hour_weekday_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"生成按小时和星期几分布的热力图时出错: {e}")

# 按日期的分享数量趋势（带移动平均线）
if 'date' in train_data.columns:
    try:
        print("生成每日分享数量趋势图...")
        # 按日期统计分享数量
        date_counts = train_data['date'].value_counts().sort_index()
        date_counts_df = pd.DataFrame({'date': date_counts.index, 'count': date_counts.values})
        
        # 计算7天移动平均
        date_counts_df['7day_ma'] = date_counts_df['count'].rolling(window=7).mean()
        

        plt.figure(figsize=(16, 8))
        
        # 使用渐变填充区域
        plt.fill_between(date_counts_df['date'], date_counts_df['count'], 
                         alpha=0.3, color='#3498db', label='每日分享数量')
        
        # 绘制原始数据线
        plt.plot(date_counts_df['date'], date_counts_df['count'], 
                 color='#3498db', alpha=0.7, linewidth=1)
        
        # 绘制移动平均线
        plt.plot(date_counts_df['date'], date_counts_df['7day_ma'], 
                 'r', linewidth=2.5, label='7天移动平均')
        
        # 标记最高点和最低点
        max_idx = date_counts_df['count'].idxmax()
        min_idx = date_counts_df['count'].idxmin()
        
        plt.scatter(date_counts_df.loc[max_idx, 'date'], date_counts_df.loc[max_idx, 'count'], 
                    s=100, c='red', marker='*', zorder=5, label='最高点')
        plt.scatter(date_counts_df.loc[min_idx, 'date'], date_counts_df.loc[min_idx, 'count'], 
                    s=100, c='blue', marker='v', zorder=5, label='最低点')
        
        # 添加标注
        plt.annotate(f"最高: {date_counts_df.loc[max_idx, 'count']:,}",
                    xy=(date_counts_df.loc[max_idx, 'date'], date_counts_df.loc[max_idx, 'count']),
                    xytext=(10, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.annotate(f"最低: {date_counts_df.loc[min_idx, 'count']:,}",
                    xy=(date_counts_df.loc[min_idx, 'date'], date_counts_df.loc[min_idx, 'count']),
                    xytext=(10, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.title('每日分享数量趋势（带7天移动平均线）', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('分享数量', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/figures/time_analysis/daily_trend_with_ma.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"生成每日分享数量趋势图时出错: {e}")

# 10. 统计每个用户的商品偏好
user_item_pref = {}
for inviter_id, group in train_data[['inviter_id', 'item_id']].groupby('inviter_id'):
    item_counts = group['item_id'].value_counts()
    user_item_pref[inviter_id] = item_counts.to_dict()

# 保存一些基本统计信息到文件
print("保存数据统计信息...")

stats = {
    '训练集大小': len(train_data),
    '测试集大小': len(test_data),
    '用户数量': len(user_info),
    '商品数量': len(item_info),
    '邀请者数量': train_data['inviter_id'].nunique(),
    '回流者数量': train_data['voter_id'].nunique(),
    '平均每个邀请者的分享次数': train_data['inviter_id'].value_counts().mean(),
    '平均每个回流者的回流次数': train_data['voter_id'].value_counts().mean(),
    '分享商品数量': train_data['item_id'].nunique(),
}

with open('results/data_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=4)

# 添加更多可视化图表
print("生成额外的数据可视化...")

# 1. 用户行为分析
# 1.1 Top 20 最活跃的邀请者

plt.figure(figsize=(12, 8))
top_inviters = train_data['inviter_id'].value_counts().nlargest(20)
ax = sns.barplot(x=top_inviters.values, y=top_inviters.index, palette=sns.color_palette("viridis", 20))
plt.title('Top 20 最活跃的邀请者', fontsize=16, fontweight='bold')
plt.xlabel('邀请次数', fontsize=12)
plt.ylabel('邀请者ID', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 添加数据标签
for i, v in enumerate(top_inviters.values):
    ax.text(v + 0.1, i, f'{v:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/user_analysis/top20_inviters.png', dpi=300, bbox_inches='tight')
plt.close()

# 1.2 Top 20 最活跃的回流者

plt.figure(figsize=(12, 8))
top_voters = train_data['voter_id'].value_counts().nlargest(20)
ax = sns.barplot(x=top_voters.values, y=top_voters.index, palette=sns.color_palette("plasma", 20))
plt.title('Top 20 最活跃的回流者', fontsize=16, fontweight='bold')
plt.xlabel('回流次数', fontsize=12)
plt.ylabel('回流者ID', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 添加数据标签
for i, v in enumerate(top_voters.values):
    ax.text(v + 0.1, i, f'{v:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/user_analysis/top20_voters.png', dpi=300, bbox_inches='tight')
plt.close()

# 1.3 不同性别用户的活跃度分布
if 'user_gender' in user_info.columns:
    # 合并用户性别信息到训练数据
    inviter_gender = pd.merge(
        train_data['inviter_id'].value_counts().reset_index().rename(columns={'inviter_id': 'user_id', 'count': 'invite_count'}),
        user_info[['user_id', 'user_gender']],
        on='user_id',
        how='left'
    )
    

    plt.figure(figsize=(10, 6))
    # 替换性别代码为可读标签
    inviter_gender['gender_label'] = inviter_gender['user_gender'].map({-1: '未知', 0: '女性', 1: '男性'})
    
    # 使用更好的颜色
    palette = {'未知': '#95a5a6', '女性': '#e74c3c', '男性': '#3498db'}
    
    # 使用小提琴图代替箱形图，更美观
    sns.violinplot(x='gender_label', y='invite_count', data=inviter_gender, 
                  palette=palette, inner='quartile', cut=0)
    
    plt.title('不同性别用户的邀请活跃度分布', fontsize=16, fontweight='bold')
    plt.xlabel('性别', fontsize=12)
    plt.ylabel('邀请次数', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加统计信息
    for i, gender in enumerate(['未知', '女性', '男性']):
        subset = inviter_gender[inviter_gender['gender_label'] == gender]
        if len(subset) > 0:
            avg = subset['invite_count'].mean()
            median = subset['invite_count'].median()
            plt.text(i, plt.ylim()[1]*0.9, f'平均: {avg:.1f}\n中位数: {median:.1f}', 
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/figures/user_analysis/gender_invite_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 添加0-10000区间的图

    plt.figure(figsize=(10, 6))
    limited_data = inviter_gender[inviter_gender['invite_count'] <= 10000]
    
    sns.violinplot(x='gender_label', y='invite_count', data=limited_data, 
                  palette=palette, inner='quartile', cut=0)
    
    plt.title('不同性别用户的邀请活跃度分布 (0-10000区间)', fontsize=16, fontweight='bold')
    plt.xlabel('性别', fontsize=12)
    plt.ylabel('邀请次数 (0-10000)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加统计信息
    for i, gender in enumerate(['未知', '女性', '男性']):
        subset = limited_data[limited_data['gender_label'] == gender]
        if len(subset) > 0:
            avg = subset['invite_count'].mean()
            median = subset['invite_count'].median()
            plt.text(i, plt.ylim()[1]*0.9, f'平均: {avg:.1f}\n中位数: {median:.1f}', 
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/figures/limited_range/gender_invite_0_10000.png', dpi=300, bbox_inches='tight')
    plt.close()

# 1.4 不同年龄段用户的活跃度分布
if 'user_age' in user_info.columns:
    # 合并用户年龄信息到训练数据
    inviter_age = pd.merge(
        train_data['inviter_id'].value_counts().reset_index().rename(columns={'inviter_id': 'user_id', 'count': 'invite_count'}),
        user_info[['user_id', 'user_age']],
        on='user_id',
        how='left'
    )
    

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='user_age', y='invite_count', data=inviter_age)
    plt.title('不同年龄段用户的邀请活跃度分布')
    plt.xlabel('年龄段')
    plt.ylabel('邀请次数')
    plt.xticks(rotation=45)
    plt.savefig('results/figures/user_analysis/age_invite_distribution.png')
    plt.close()
    
    # 添加0-10000区间的图

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='user_age', y='invite_count', data=inviter_age[inviter_age['invite_count'] <= 10000])
    plt.title('不同年龄段用户的邀请活跃度分布 (0-10000区间)')
    plt.xlabel('年龄段')
    plt.ylabel('邀请次数 (0-10000)')
    plt.xticks(rotation=45)
    plt.savefig('results/figures/limited_range/age_invite_0_10000.png')
    plt.close()

# 1.5 不同等级用户的活跃度分布
if 'user_level' in user_info.columns:
    # 合并用户等级信息到训练数据
    inviter_level = pd.merge(
        train_data['inviter_id'].value_counts().reset_index().rename(columns={'inviter_id': 'user_id', 'count': 'invite_count'}),
        user_info[['user_id', 'user_level']],
        on='user_id',
        how='left'
    )
    

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='user_level', y='invite_count', data=inviter_level)
    plt.title('不同等级用户的邀请活跃度分布')
    plt.xlabel('用户等级')
    plt.ylabel('邀请次数')
    plt.savefig('results/figures/user_analysis/level_invite_distribution.png')
    plt.close()
    
    # 添加0-10000区间的图

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='user_level', y='invite_count', data=inviter_level[inviter_level['invite_count'] <= 10000])
    plt.title('不同等级用户的邀请活跃度分布 (0-10000区间)')
    plt.xlabel('用户等级')
    plt.ylabel('邀请次数 (0-10000)')
    plt.savefig('results/figures/limited_range/level_invite_0_10000.png')
    plt.close()

# 2. 商品特征分析
# 2.1 Top 20 最受欢迎的商品

plt.figure(figsize=(12, 8))
top_items = train_data['item_id'].value_counts().nlargest(20)
ax = sns.barplot(x=top_items.values, y=top_items.index, palette=sns.color_palette("YlOrRd", 20))
plt.title('Top 20 最受欢迎的商品', fontsize=16, fontweight='bold')
plt.xlabel('分享次数', fontsize=12)
plt.ylabel('商品ID', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 添加数据标签
for i, v in enumerate(top_items.values):
    ax.text(v + 0.1, i, f'{v:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/item_analysis/top20_items.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.2 不同类别商品的分享次数分布
if 'cate_id' in item_info.columns:
    # 合并商品类别信息到训练数据
    item_category = pd.merge(
        train_data['item_id'].value_counts().reset_index().rename(columns={'item_id': 'item_id', 'count': 'share_count'}),
        item_info[['item_id', 'cate_id']],
        on='item_id',
        how='left'
    )
    
    # 按类别统计平均分享次数
    category_avg_shares = item_category.groupby('cate_id')['share_count'].mean().nlargest(20)
    

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=category_avg_shares.index, y=category_avg_shares.values, 
                    palette=sns.color_palette("Blues_r", len(category_avg_shares)))
    
    plt.title('不同类别商品的平均分享次数 (Top 20)', fontsize=16, fontweight='bold')
    plt.xlabel('商品类别ID', fontsize=12)
    plt.ylabel('平均分享次数', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数据标签
    for i, v in enumerate(category_avg_shares.values):
        ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=9, rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/figures/item_analysis/category_avg_shares.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 添加箱线图展示分布

    plt.figure(figsize=(16, 8))
    top_categories = item_category['cate_id'].value_counts().nlargest(10).index
    category_data = item_category[item_category['cate_id'].isin(top_categories)]
    
    sns.boxplot(x='cate_id', y='share_count', data=category_data, 
               palette=sns.color_palette("Blues_r", len(top_categories)))
    
    plt.title('Top 10 类别商品的分享次数分布', fontsize=16, fontweight='bold')
    plt.xlabel('商品类别ID', fontsize=12)
    plt.ylabel('分享次数', fontsize=12)
    plt.yscale('log')  # 使用对数刻度更好地展示分布
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/figures/item_analysis/category_share_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2.3 不同一级类目商品的分享次数分布
if 'cate_level1_id' in item_info.columns:
    # 合并商品一级类目信息到训练数据
    item_category1 = pd.merge(
        train_data['item_id'].value_counts().reset_index().rename(columns={'item_id': 'item_id', 'count': 'share_count'}),
        item_info[['item_id', 'cate_level1_id']],
        on='item_id',
        how='left'
    )
    
    # 按一级类目统计平均分享次数
    category1_avg_shares = item_category1.groupby('cate_level1_id')['share_count'].mean().nlargest(20)
    

    plt.figure(figsize=(14, 8))
    sns.barplot(x=category1_avg_shares.index, y=category1_avg_shares.values)
    plt.title('不同一级类目商品的平均分享次数 (Top 20)')
    plt.xlabel('一级类目ID')
    plt.ylabel('平均分享次数')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('results/figures/item_analysis/category1_avg_shares.png')
    plt.close()

# 2.4 不同品牌商品的分享次数分布
if 'brand_id' in item_info.columns:
    # 合并商品品牌信息到训练数据
    item_brand = pd.merge(
        train_data['item_id'].value_counts().reset_index().rename(columns={'item_id': 'item_id', 'count': 'share_count'}),
        item_info[['item_id', 'brand_id']],
        on='item_id',
        how='left'
    )
    
    # 按品牌统计平均分享次数
    brand_avg_shares = item_brand.groupby('brand_id')['share_count'].mean().nlargest(20)
    

    plt.figure(figsize=(14, 8))
    sns.barplot(x=brand_avg_shares.index, y=brand_avg_shares.values)
    plt.title('不同品牌商品的平均分享次数 (Top 20)')
    plt.xlabel('品牌ID')
    plt.ylabel('平均分享次数')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('results/figures/item_analysis/brand_avg_shares.png')
    plt.close()

# 4. 更多交互网络分析
# 4.1 邀请者-回流者关系网络可视化（取样本）
try:
    print("生成邀请者-回流者关系网络可视化...")
    G = nx.DiGraph()
    
    # 添加边（限制样本数量以防止内存溢出）
    edge_sample = train_data[['inviter_id', 'voter_id']].sample(n=min(1000, len(train_data)))
    edges = list(zip(edge_sample['inviter_id'], edge_sample['voter_id']))
    G.add_edges_from(edges)
    
    # 计算节点度数
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    total_degrees = {node: in_degrees.get(node, 0) + out_degrees.get(node, 0) for node in set(in_degrees) | set(out_degrees)}
    
    # 节点大小基于度数，使用对数缩放以更好地显示
    node_size = [max(20, np.log1p(total_degrees.get(node, 0)) * 30) for node in G.nodes()]
    
    # 使用更好的布局算法

    plt.figure(figsize=(20, 20), facecolor='white')
    
    # 使用更好的布局算法
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # 根据入度和出度设置节点颜色
    node_color = []
    for node in G.nodes():
        in_deg = in_degrees.get(node, 0)
        out_deg = out_degrees.get(node, 0)
        if in_deg > out_deg:
            # 更多入度，使用蓝色系
            ratio = min(1.0, in_deg / (in_deg + out_deg + 0.1))
            node_color.append(plt.cm.Blues(0.5 + ratio/2))
        else:
            # 更多出度，使用红色系
            ratio = min(1.0, out_deg / (in_deg + out_deg + 0.1))
            node_color.append(plt.cm.Reds(0.5 + ratio/2))
    
    # 绘制边，使用透明度和曲线
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.6, 
                          edge_color='gray', arrows=True,
                          arrowstyle='-|>', arrowsize=10,
                          connectionstyle='arc3,rad=0.1')
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                          node_color=node_color, alpha=0.8,
                          edgecolors='white', linewidths=0.5)
    
    # 标记高度数节点
    top_nodes = sorted(G.nodes(), key=lambda x: total_degrees.get(x, 0), reverse=True)[:10]
    nx.draw_networkx_labels(G, pos, {node: str(node) for node in top_nodes}, 
                           font_size=10, font_color='black',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # 添加图例
    plt.plot([0], [0], 'o', markersize=10, color=plt.cm.Blues(0.8), label='主要接收者')
    plt.plot([0], [0], 'o', markersize=10, color=plt.cm.Reds(0.8), label='主要发送者')
    plt.legend(loc='lower right', fontsize=12)
    
    plt.title('邀请者-回流者关系网络（样本）', fontsize=20, fontweight='bold', pad=20)
    plt.text(0.5, 0.02, f'节点数: {len(G.nodes())}, 连接数: {len(G.edges())}', 
             ha='center', transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/figures/network_analysis/inviter_voter_network.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成邀请者-回流者关系网络可视化时出错: {e}")
    # 跳过这个可视化，继续执行后面的代码

# 4.2 邀请者-商品二部图（取样本）
try:
    print("生成邀请者-商品二部图...")
    B = nx.Graph()
    
    # 首先采样数据
    sample_data = train_data.sample(n=min(1000, len(train_data)))
    
    # 添加节点和边
    inviters = set(sample_data['inviter_id'])
    items = set(sample_data['item_id'])
    
    # 添加所有节点
    B.add_nodes_from(inviters, bipartite=0)  # 邀请者节点
    B.add_nodes_from(items, bipartite=1)     # 商品节点
    
    # 添加边
    edges = list(zip(sample_data['inviter_id'], sample_data['item_id']))
    B.add_edges_from(edges)
    
    # 使用更好的布局算法

    plt.figure(figsize=(20, 16), facecolor='white')
    
    # 使用二部图专用布局
    pos = nx.spring_layout(B, k=0.5, iterations=50, seed=42)
    
    # 计算节点度数
    degrees = dict(B.degree())
    
    # 邀请者和商品的节点大小基于度数
    inviter_sizes = [max(30, np.log1p(degrees.get(node, 0)) * 20) for node in inviters]
    item_sizes = [max(20, np.log1p(degrees.get(node, 0)) * 15) for node in items]
    
    # 绘制边，使用透明度
    nx.draw_networkx_edges(B, pos, width=0.5, alpha=0.2, edge_color='#95a5a6')
    
    # 绘制邀请者节点，使用渐变色
    inviter_degrees = {node: degrees.get(node, 0) for node in inviters}
    max_inviter_degree = max(inviter_degrees.values()) if inviter_degrees else 1
    inviter_colors = [plt.cm.viridis(0.1 + 0.9 * degrees.get(node, 0) / max_inviter_degree) for node in inviters]
    
    nx.draw_networkx_nodes(B, pos, nodelist=list(inviters), 
                          node_color=inviter_colors, 
                          node_size=inviter_sizes, 
                          alpha=0.8, label='邀请者',
                          edgecolors='white', linewidths=0.5)
    
    # 绘制商品节点，使用渐变色
    item_degrees = {node: degrees.get(node, 0) for node in items}
    max_item_degree = max(item_degrees.values()) if item_degrees else 1
    item_colors = [plt.cm.plasma(0.1 + 0.9 * degrees.get(node, 0) / max_item_degree) for node in items]
    
    nx.draw_networkx_nodes(B, pos, nodelist=list(items), 
                          node_color=item_colors, 
                          node_size=item_sizes, 
                          alpha=0.8, label='商品',
                          edgecolors='white', linewidths=0.5)
    
    # 标记一些高度数节点
    top_inviters = sorted(inviters, key=lambda x: degrees.get(x, 0), reverse=True)[:5]
    top_items = sorted(items, key=lambda x: degrees.get(x, 0), reverse=True)[:5]
    
    # 标记顶级邀请者
    nx.draw_networkx_labels(B, pos, {node: f"用户: {node}" for node in top_inviters}, 
                           font_size=10, font_color='black',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # 标记顶级商品
    nx.draw_networkx_labels(B, pos, {node: f"商品: {node}" for node in top_items}, 
                           font_size=10, font_color='black',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    plt.title('邀请者-商品二部图（样本）', fontsize=20, fontweight='bold', pad=20)
    plt.text(0.5, 0.02, f'邀请者: {len(inviters)}, 商品: {len(items)}, 连接: {len(edges)}', 
             ha='center', transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # 添加图例
    plt.legend(loc='lower right', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/figures/network_analysis/inviter_item_bipartite.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成邀请者-商品二部图时出错: {e}")
    # 跳过这个可视化，继续执行后面的代码

print("数据预处理和可视化完成！结果保存在 results 文件夹中。")

# 添加更多高级可视化图表
print("生成额外的高级可视化图表...")

# 创建额外的可视化文件夹
if not os.path.exists('results/figures/advanced'):
    os.makedirs('results/figures/advanced')
if not os.path.exists('results/figures/correlation'):
    os.makedirs('results/figures/correlation')
if not os.path.exists('results/figures/distribution'):
    os.makedirs('results/figures/distribution')

# 1. 用户行为时间热力图 - 按小时和日期的分布
try:
    print("生成用户行为时间热力图...")
    
    # 提取日期和小时
    train_data['date_str'] = train_data['timestamp'].dt.strftime('%m-%d')
    train_data['hour'] = train_data['timestamp'].dt.hour
    
    # 创建日期和小时的交叉表
    date_hour_counts = pd.crosstab(train_data['date_str'], train_data['hour'])
    
    # 仅选择有足够数据的日期（最多30天）
    top_dates = train_data['date_str'].value_counts().nlargest(30).index
    date_hour_subset = date_hour_counts.loc[top_dates]
    
    plt.figure(figsize=(18, 10))
    
    # 使用更好的颜色映射
    cmap = plt.cm.YlOrRd
    
    # 创建热力图
    ax = sns.heatmap(date_hour_subset, cmap=cmap, linewidths=0.1, 
                    cbar_kws={'label': '分享数量'})
    
    plt.title('用户行为时间热力图 (日期 × 小时)', fontsize=16, fontweight='bold')
    plt.xlabel('小时', fontsize=12)
    plt.ylabel('日期', fontsize=12)
    
    # 找出热点
    max_val = date_hour_subset.max().max()
    max_date = date_hour_subset.max(axis=1).idxmax()
    max_hour = date_hour_subset.max().idxmax()
    
    plt.text(0.5, -0.07, f'热点时段: {max_date} {max_hour}时 (分享数量: {max_val})', 
             ha='center', transform=ax.transAxes, fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('results/figures/advanced/user_behavior_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成用户行为时间热力图时出错: {e}")

# 2. 用户性别与年龄的交叉分析
try:
    print("生成用户性别与年龄的交叉分析图...")
    
    # 合并用户信息
    user_demo = user_info[['user_id', 'user_gender', 'user_age']].copy()
    user_demo['gender'] = user_demo['user_gender'].map({-1: '未知', 0: '女性', 1: '男性'})
    user_demo['age'] = user_demo['user_age'].map(lambda x: '未知' if x == -1 else f'年龄段{x}')
    
    # 创建交叉表
    gender_age_counts = pd.crosstab(user_demo['gender'], user_demo['age'])
    
    plt.figure(figsize=(14, 8))
    
    # 绘制堆叠条形图
    gender_age_counts.plot(kind='bar', stacked=True, colormap='viridis')
    
    plt.title('用户性别与年龄分布', fontsize=16, fontweight='bold')
    plt.xlabel('性别', fontsize=12)
    plt.ylabel('用户数量', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='年龄段')
    
    plt.tight_layout()
    plt.savefig('results/figures/advanced/gender_age_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 百分比堆叠图
    plt.figure(figsize=(14, 8))
    
    # 计算百分比
    gender_age_pct = gender_age_counts.div(gender_age_counts.sum(axis=1), axis=0) * 100
    
    # 绘制百分比堆叠条形图
    gender_age_pct.plot(kind='bar', stacked=True, colormap='viridis')
    
    plt.title('各性别用户的年龄段分布 (%)', fontsize=16, fontweight='bold')
    plt.xlabel('性别', fontsize=12)
    plt.ylabel('百分比 (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='年龄段')
    
    plt.tight_layout()
    plt.savefig('results/figures/advanced/gender_age_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成用户性别与年龄的交叉分析图时出错: {e}")

# 3. 用户等级与活跃度的关系
try:
    print("生成用户等级与活跃度关系图...")
    
    # 计算每个用户的活跃度（分享次数）
    user_activity = train_data['inviter_id'].value_counts().reset_index()
    user_activity.columns = ['user_id', 'activity_count']
    
    # 合并用户等级信息
    user_level_activity = pd.merge(
        user_activity,
        user_info[['user_id', 'user_level']],
        on='user_id',
        how='left'
    )
    
    plt.figure(figsize=(12, 8))
    
    # 使用小提琴图展示分布
    sns.violinplot(x='user_level', y='activity_count', data=user_level_activity, 
                  palette='viridis', inner='quartile', cut=0)
    
    plt.title('用户等级与活跃度关系', fontsize=16, fontweight='bold')
    plt.xlabel('用户等级', fontsize=12)
    plt.ylabel('活跃度（分享次数）', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加对数刻度以便更好地查看分布
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('results/figures/advanced/user_level_activity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 添加箱线图
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='user_level', y='activity_count', data=user_level_activity, palette='viridis')
    
    plt.title('用户等级与活跃度关系 (箱线图)', fontsize=16, fontweight='bold')
    plt.xlabel('用户等级', fontsize=12)
    plt.ylabel('活跃度（分享次数）', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/figures/advanced/user_level_activity_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成用户等级与活跃度关系图时出错: {e}")

# 4. 商品类别的流行度趋势（按时间）
try:
    print("生成商品类别流行度趋势图...")
    
    # 合并商品类别信息
    item_share_category = pd.merge(
        train_data[['item_id', 'timestamp', 'date']],
        item_info[['item_id', 'cate_id']],
        on='item_id',
        how='left'
    )
    
    # 获取前5个最流行的类别
    top_categories = item_share_category['cate_id'].value_counts().nlargest(5).index
    
    # 过滤数据
    top_category_data = item_share_category[item_share_category['cate_id'].isin(top_categories)]
    
    # 按日期和类别统计分享次数
    category_trend = top_category_data.groupby(['date', 'cate_id']).size().reset_index(name='count')
    
    plt.figure(figsize=(16, 8))
    
    # 为每个类别绘制一条线
    for i, category in enumerate(top_categories):
        cat_data = category_trend[category_trend['cate_id'] == category]
        plt.plot(cat_data['date'], cat_data['count'], 
                marker='o', markersize=4, linewidth=2, 
                label=f'类别 {category}')
    
    plt.title('热门商品类别的流行度趋势', fontsize=16, fontweight='bold')
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('分享次数', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='商品类别')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/figures/advanced/category_popularity_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成商品类别流行度趋势图时出错: {e}")

# 5. 用户相似度网络（基于共同分享的商品）
try:
    print("生成用户相似度网络图...")
    
    # 创建用户-商品矩阵 - 减少数据规模以避免内存错误
    # 选择最活跃的用户和最热门的商品
    top_users = train_data['inviter_id'].value_counts().nlargest(50).index
    top_items = train_data['item_id'].value_counts().nlargest(100).index
    
    # 过滤数据
    filtered_data = train_data[
        (train_data['inviter_id'].isin(top_users)) & 
        (train_data['item_id'].isin(top_items))
    ]
    
    # 创建用户-商品矩阵
    user_item_matrix = pd.crosstab(filtered_data['inviter_id'], filtered_data['item_id'])
    
    # 计算用户间的余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    user_similarity = cosine_similarity(user_item_matrix)
    
    # 创建相似度DataFrame
    similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    # 创建网络图
    G = nx.Graph()
    
    # 添加节点
    for user in user_item_matrix.index:
        G.add_node(user)
    
    # 添加边（仅添加相似度高于阈值的边）
    threshold = 0.3
    for i, user1 in enumerate(user_item_matrix.index):
        for j, user2 in enumerate(user_item_matrix.index):
            if i < j:  # 避免重复
                similarity = similarity_df.loc[user1, user2]
                if similarity > threshold:
                    G.add_edge(user1, user2, weight=similarity)
    
    plt.figure(figsize=(16, 16))
    
    # 设置布局
    pos = nx.spring_layout(G, k=0.3, seed=42)
    
    # 获取边权重
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    
    # 获取节点度数
    node_degrees = dict(G.degree())
    node_sizes = [max(100, v * 20) for v in node_degrees.values()]
    
    # 绘制网络
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color=list(node_degrees.values()), 
                          cmap=plt.cm.viridis, alpha=0.8)
    
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
    
    plt.title('用户相似度网络（基于共同分享的商品）', fontsize=16, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/figures/advanced/user_similarity_network.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成用户相似度网络图时出错: {e}")

# 6. 商品属性相关性分析
try:
    print("生成商品属性相关性分析图...")
    
    # 合并商品信息和分享次数
    item_share_count = train_data['item_id'].value_counts().reset_index()
    item_share_count.columns = ['item_id', 'share_count']
    
    item_analysis = pd.merge(
        item_share_count,
        item_info,
        on='item_id',
        how='left'
    )
    
    # 计算每个类别和品牌的平均分享次数
    category_avg = item_analysis.groupby('cate_id')['share_count'].mean().reset_index()
    category_avg.columns = ['cate_id', 'avg_shares']
    category_avg = category_avg.sort_values('avg_shares', ascending=False)
    
    brand_avg = item_analysis.groupby('brand_id')['share_count'].mean().reset_index()
    brand_avg.columns = ['brand_id', 'avg_shares']
    brand_avg = brand_avg.sort_values('avg_shares', ascending=False)
    
    # 绘制类别平均分享次数
    plt.figure(figsize=(14, 8))
    
    top_n = min(20, len(category_avg))
    ax = sns.barplot(x='cate_id', y='avg_shares', data=category_avg.head(top_n), palette='viridis')
    
    plt.title('商品类别的平均分享次数 (Top 20)', fontsize=16, fontweight='bold')
    plt.xlabel('商品类别', fontsize=12)
    plt.ylabel('平均分享次数', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=90)
    
    # 添加数据标签
    for i, v in enumerate(category_avg.head(top_n)['avg_shares']):
        ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/figures/advanced/category_avg_shares_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制品牌平均分享次数
    plt.figure(figsize=(14, 8))
    
    top_n = min(20, len(brand_avg))
    ax = sns.barplot(x='brand_id', y='avg_shares', data=brand_avg.head(top_n), palette='plasma')
    
    plt.title('商品品牌的平均分享次数 (Top 20)', fontsize=16, fontweight='bold')
    plt.xlabel('商品品牌', fontsize=12)
    plt.ylabel('平均分享次数', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=90)
    
    # 添加数据标签
    for i, v in enumerate(brand_avg.head(top_n)['avg_shares']):
        ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/figures/advanced/brand_avg_shares_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成商品属性相关性分析图时出错: {e}")

# 7. 用户行为模式分析 - 分享时间间隔
try:
    print("生成用户行为模式分析图...")
    
    # 选择活跃用户
    active_users = train_data['inviter_id'].value_counts().nlargest(100).index
    active_user_data = train_data[train_data['inviter_id'].isin(active_users)]
    
    # 按用户和时间戳排序
    active_user_data = active_user_data.sort_values(['inviter_id', 'timestamp'])
    
    # 计算每个用户的分享时间间隔
    active_user_data['prev_timestamp'] = active_user_data.groupby('inviter_id')['timestamp'].shift(1)
    active_user_data['time_diff'] = (active_user_data['timestamp'] - active_user_data['prev_timestamp']).dt.total_seconds() / 3600  # 转换为小时
    
    # 过滤有效的时间差
    valid_diffs = active_user_data.dropna(subset=['time_diff'])
    
    # 绘制时间间隔分布
    plt.figure(figsize=(12, 8))
    
    # 限制范围以便更好地可视化
    max_hours = 72  # 3天
    filtered_diffs = valid_diffs[valid_diffs['time_diff'] <= max_hours]
    
    sns.histplot(filtered_diffs['time_diff'], bins=50, kde=True, color='#3498db')
    
    plt.title('用户分享行为的时间间隔分布', fontsize=16, fontweight='bold')
    plt.xlabel('时间间隔（小时）', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加垂直线标记重要时间点
    plt.axvline(x=24, color='r', linestyle='--', alpha=0.7, label='24小时')
    plt.axvline(x=48, color='g', linestyle='--', alpha=0.7, label='48小时')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/figures/advanced/sharing_time_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制对数刻度的时间间隔分布
    plt.figure(figsize=(12, 8))
    
    # 使用对数变换
    valid_diffs['log_time_diff'] = np.log1p(valid_diffs['time_diff'])
    
    sns.histplot(valid_diffs['log_time_diff'], bins=50, kde=True, color='#2ecc71')
    
    plt.title('用户分享行为的时间间隔分布（对数刻度）', fontsize=16, fontweight='bold')
    plt.xlabel('时间间隔（对数小时）', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/figures/advanced/sharing_time_intervals_log.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成用户行为模式分析图时出错: {e}")

# 8. 用户回流分析
try:
    print("生成用户回流分析图...")
    
    # 计算每个邀请者的回流率
    inviter_stats = train_data.groupby('inviter_id').agg(
        total_shares=('item_id', 'count'),
        unique_voters=('voter_id', 'nunique')
    ).reset_index()
    
    inviter_stats['return_rate'] = inviter_stats['unique_voters'] / inviter_stats['total_shares']
    
    # 按总分享次数分组
    def get_share_group(count):
        if count < 10:
            return '<10'
        elif count < 50:
            return '10-50'
        elif count < 100:
            return '50-100'
        elif count < 500:
            return '100-500'
        else:
            return '>500'
    
    inviter_stats['share_group'] = inviter_stats['total_shares'].apply(get_share_group)
    
    # 计算每个组的平均回流率
    group_stats = inviter_stats.groupby('share_group').agg(
        avg_return_rate=('return_rate', 'mean'),
        count=('inviter_id', 'count')
    ).reset_index()
    
    # 确保组按正确顺序排列
    group_order = ['<10', '10-50', '50-100', '100-500', '>500']
    group_stats['share_group'] = pd.Categorical(group_stats['share_group'], categories=group_order, ordered=True)
    group_stats = group_stats.sort_values('share_group')
    
    plt.figure(figsize=(12, 8))
    
    # 创建双轴图
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 绘制回流率条形图
    bars = ax1.bar(group_stats['share_group'], group_stats['avg_return_rate'], color='#3498db', alpha=0.7)
    ax1.set_xlabel('分享次数组', fontsize=12)
    ax1.set_ylabel('平均回流率', fontsize=12, color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', color='#3498db')
    
    # 创建第二个y轴
    ax2 = ax1.twinx()
    
    # 绘制用户数量线图
    ax2.plot(group_stats['share_group'], group_stats['count'], 'o-', color='#e74c3c', linewidth=2)
    ax2.set_ylabel('用户数量', fontsize=12, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # 添加数据标签
    for i, v in enumerate(group_stats['count']):
        ax2.text(i, v + 0.1, f'{v:,}', ha='center', va='bottom', color='#e74c3c')
    
    plt.title('不同活跃度用户的平均回流率', fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.savefig('results/figures/advanced/return_rate_by_activity.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成用户回流分析图时出错: {e}")

# 9. 用户与商品的双向聚类图
try:
    print("生成用户与商品的双向聚类图...")
    
    # 选择活跃用户和热门商品
    top_users = train_data['inviter_id'].value_counts().nlargest(30).index
    top_items = train_data['item_id'].value_counts().nlargest(30).index
    
    # 过滤数据
    filtered_data = train_data[
        (train_data['inviter_id'].isin(top_users)) & 
        (train_data['item_id'].isin(top_items))
    ]
    
    # 创建用户-商品矩阵
    user_item_matrix = pd.crosstab(filtered_data['inviter_id'], filtered_data['item_id'])
    
    # 使用对数变换使数据更易于可视化
    user_item_matrix = np.log1p(user_item_matrix)
    
    plt.figure(figsize=(16, 12))
    
    # 创建聚类热图
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.clustermap(user_item_matrix, cmap=cmap, figsize=(16, 12),
                  row_cluster=True, col_cluster=True,
                  linewidths=0.1, xticklabels=1, yticklabels=1)
    
    plt.savefig('results/figures/advanced/user_item_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成用户与商品的双向聚类图时出错: {e}")

# 10. 商品分享与回流的相关性分析
try:
    print("生成商品分享与回流的相关性分析图...")
    
    # 计算每个商品的分享次数和回流用户数
    item_stats = train_data.groupby('item_id').agg(
        shares=('inviter_id', 'count'),
        unique_voters=('voter_id', 'nunique')
    ).reset_index()
    
    # 计算回流率
    item_stats['return_rate'] = item_stats['unique_voters'] / item_stats['shares']
    
    # 过滤掉异常值
    filtered_items = item_stats[item_stats['shares'] >= 5]  # 至少有5次分享
    
    plt.figure(figsize=(12, 8))
    
    # 绘制散点图
    sns.scatterplot(x='shares', y='unique_voters', data=filtered_items, 
                   alpha=0.6, s=50, hue='return_rate', palette='viridis')
    
    plt.title('商品分享次数与回流用户数的关系', fontsize=16, fontweight='bold')
    plt.xlabel('分享次数', fontsize=12)
    plt.ylabel('回流用户数', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加回归线
    sns.regplot(x='shares', y='unique_voters', data=filtered_items, 
               scatter=False, ci=None, color='red')
    
    plt.tight_layout()
    plt.savefig('results/figures/advanced/item_share_return_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 对数刻度版本
    plt.figure(figsize=(12, 8))
    
    # 先转换为对数
    filtered_items['log_shares'] = np.log1p(filtered_items['shares'])
    filtered_items['log_voters'] = np.log1p(filtered_items['unique_voters'])
    
    # 绘制散点图（使用已转换的数据）
    sns.scatterplot(x='log_shares', y='log_voters', data=filtered_items, 
                   alpha=0.6, s=50, hue='return_rate', palette='viridis')
    
    plt.title('商品分享次数与回流用户数的关系（对数刻度）', fontsize=16, fontweight='bold')
    plt.xlabel('分享次数（对数）', fontsize=12)
    plt.ylabel('回流用户数（对数）', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加回归线（使用已转换的数据）
    sns.regplot(x='log_shares', y='log_voters', data=filtered_items, 
               scatter=False, ci=None, color='red')
    
    plt.tight_layout()
    plt.savefig('results/figures/advanced/item_share_return_correlation_log.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"生成商品分享与回流的相关性分析图时出错: {e}")

print("额外的高级可视化图表生成完成！") 