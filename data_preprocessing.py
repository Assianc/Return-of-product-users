import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

# 数据路径
TRAIN_PATH = 'train/preliminary'
ITEM_SHARE_FILE = os.path.join(TRAIN_PATH, 'item_share_train_info.json')
USER_INFO_FILE = os.path.join(TRAIN_PATH, 'user_info.json')
ITEM_INFO_FILE = os.path.join(TRAIN_PATH, 'item_info.json')

def load_json_data(file_path):
    """加载JSON格式的数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def convert_to_dataframe(data, data_type):
    """将JSON数据转换为DataFrame"""
    if data_type == 'item_share':
        df = pd.DataFrame(data)
        # 将timestamp转换为datetime格式
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    elif data_type == 'user_info':
        return pd.DataFrame(data)
    elif data_type == 'item_info':
        return pd.DataFrame(data)
    else:
        raise ValueError("Unknown data type")

def preprocess_data():
    """数据预处理主函数"""
    print("加载数据...")
    item_share_data = load_json_data(ITEM_SHARE_FILE)
    user_info_data = load_json_data(USER_INFO_FILE)
    item_info_data = load_json_data(ITEM_INFO_FILE)
    
    print("转换数据格式...")
    item_share_df = convert_to_dataframe(item_share_data, 'item_share')
    user_info_df = convert_to_dataframe(user_info_data, 'user_info')
    item_info_df = convert_to_dataframe(item_info_data, 'item_info')
    
    # 设置索引，方便后续合并
    user_info_df.set_index('user_id', inplace=True)
    item_info_df.set_index('item_id', inplace=True)
    
    print("数据基本信息:")
    print(f"商品分享记录数量: {len(item_share_df)}")
    print(f"用户数量: {len(user_info_df)}")
    print(f"商品数量: {len(item_info_df)}")
    
    # 检查缺失值
    print("\n检查缺失值:")
    print(f"商品分享数据缺失值:\n{item_share_df.isnull().sum()}")
    print(f"用户信息数据缺失值:\n{user_info_df.isnull().sum()}")
    print(f"商品信息数据缺失值:\n{item_info_df.isnull().sum()}")
    
    # 返回处理后的数据
    return item_share_df, user_info_df, item_info_df

def split_data(item_share_df, test_size=0.2, random_state=42):
    """分割数据集为训练集和测试集"""
    # 按时间排序
    item_share_df = item_share_df.sort_values('timestamp')
    
    # 获取唯一的(inviter_id, item_id, timestamp)组合
    unique_combinations = item_share_df[['inviter_id', 'item_id', 'timestamp']].drop_duplicates()
    
    # 使用时间划分，最后test_size比例的数据作为测试集
    split_idx = int(len(unique_combinations) * (1 - test_size))
    train_combinations = unique_combinations.iloc[:split_idx]
    test_combinations = unique_combinations.iloc[split_idx:]
    
    # 合并回原始数据
    train_df = pd.merge(train_combinations, item_share_df, 
                        on=['inviter_id', 'item_id', 'timestamp'])
    test_df = pd.merge(test_combinations, item_share_df, 
                       on=['inviter_id', 'item_id', 'timestamp'])
    
    print(f"\n数据集划分完成:")
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    
    return train_df, test_df

if __name__ == "__main__":
    # 执行数据预处理
    item_share_df, user_info_df, item_info_df = preprocess_data()
    
    # 划分训练集和测试集
    train_df, test_df = split_data(item_share_df)
    
    # 保存处理后的数据（可选）
    train_df.to_csv('processed_data/train.csv', index=False)
    test_df.to_csv('processed_data/test.csv', index=False)
    user_info_df.to_csv('processed_data/user_info.csv')
    item_info_df.to_csv('processed_data/item_info.csv') 