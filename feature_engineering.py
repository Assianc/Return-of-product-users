import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder
import networkx as nx

# 创建结果文件夹
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/features'):
    os.makedirs('results/features')

print("开始特征工程...")

# 从文件读取数据
def read_json_to_df(file_path, chunk_size=None):
    if chunk_size:
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk = []
            for i, line in tqdm(enumerate(f), desc=f"Reading {file_path}"):
                try:
                    chunk.append(json.loads(line))
                    if (i + 1) % chunk_size == 0:
                        chunks.append(pd.DataFrame(chunk))
                        chunk = []
                except Exception as e:
                    print(f"Error parsing line {i}: {e}")
            if chunk:  # 添加最后一个不完整的chunk
                chunks.append(pd.DataFrame(chunk))
        
        if not chunks:
            return pd.DataFrame()
        
        return pd.concat(chunks, ignore_index=True)
    else:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc=f"Reading {file_path}")):
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(f"Error parsing line {i}: {e}")
        return pd.DataFrame(data)

# 加载训练集数据
print("加载数据...")
train_data = read_json_to_df('data/train/item_share_train_info.json', chunk_size=200000)
user_info = read_json_to_df('data/train/user_info.json')
item_info = read_json_to_df('data/train/item_info.json', chunk_size=100000)
test_data = read_json_to_df('data/test/item_share_preliminary_test_info.json')

# 转换时间戳为datetime类型
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])

# 添加triple_id到测试集
test_data['triple_id'] = test_data.index.astype(str)

# 特征工程
print("生成特征...")

# 1. 构建用户-商品交互矩阵
print("构建用户-商品交互矩阵...")
user_item_matrix = train_data.groupby(['inviter_id', 'item_id']).size().reset_index(name='interaction_count')

# 2. 构建用户-用户交互网络
print("构建用户-用户交互网络...")
user_user_matrix = train_data.groupby(['inviter_id', 'voter_id']).size().reset_index(name='interaction_count')

# 3. 提取时间特征
train_data['hour'] = train_data['timestamp'].dt.hour
train_data['day_of_week'] = train_data['timestamp'].dt.dayofweek
train_data['month'] = train_data['timestamp'].dt.month
train_data['day'] = train_data['timestamp'].dt.day

# 对测试集也提取同样的时间特征
test_data['hour'] = test_data['timestamp'].dt.hour
test_data['day_of_week'] = test_data['timestamp'].dt.dayofweek
test_data['month'] = test_data['timestamp'].dt.month
test_data['day'] = test_data['timestamp'].dt.day

# 4. 用户活跃度特征
print("计算用户活跃度特征...")
user_invites = train_data['inviter_id'].value_counts().reset_index()
user_invites.columns = ['user_id', 'invite_count']

user_votes = train_data['voter_id'].value_counts().reset_index()
user_votes.columns = ['user_id', 'voted_count']

user_activity = pd.merge(user_invites, user_votes, on='user_id', how='outer').fillna(0)

# 5. 商品流行度特征
print("计算商品流行度特征...")
item_popularity = train_data['item_id'].value_counts().reset_index()
item_popularity.columns = ['item_id', 'item_popularity']

# 6. 社交网络特征
print("生成社交网络特征...")
# 构建图
G = nx.DiGraph()
edges = list(zip(train_data['inviter_id'], train_data['voter_id']))
G.add_edges_from(edges)

# 计算每个用户的中心性指标
# 通过抽样计算中心性指标，避免内存溢出
sample_nodes = list(G.nodes())
if len(sample_nodes) > 10000:
    sample_nodes = np.random.choice(sample_nodes, 10000, replace=False)

print("计算度中心性...")
in_degree = {node: val for node, val in G.in_degree(sample_nodes)}
out_degree = {node: val for node, val in G.out_degree(sample_nodes)}

print("计算介数中心性（可能较慢）...")
# 抽取一个小子图计算介数中心性（完整计算太慢）
subgraph_nodes = np.random.choice(list(G.nodes()), min(1000, len(G.nodes())), replace=False)
subgraph = G.subgraph(subgraph_nodes)
betweenness = nx.betweenness_centrality(subgraph)

# 保存网络特征
network_features = pd.DataFrame({
    'user_id': list(in_degree.keys()),
    'in_degree': list(in_degree.values()),
    'out_degree': list(out_degree.values())
})

# 将介数中心性添加到网络特征中
betweenness_df = pd.DataFrame({
    'user_id': list(betweenness.keys()),
    'betweenness': list(betweenness.values())
})
network_features = pd.merge(network_features, betweenness_df, on='user_id', how='left')

# 7. 用户-商品交互频次特征
print("计算用户-商品交互频次...")
user_item_freq = train_data.groupby(['inviter_id', 'item_id']).size().reset_index(name='interaction_freq')

# 8. 用户-类别交互频次特征
print("计算用户-类别交互频次...")
train_with_item = pd.merge(train_data, item_info, on='item_id', how='left')
user_cate_freq = train_with_item.groupby(['inviter_id', 'cate_id']).size().reset_index(name='cate_interaction_freq')
user_cate1_freq = train_with_item.groupby(['inviter_id', 'cate_level1_id']).size().reset_index(name='cate1_interaction_freq')

# 9. 用户相似度特征
print("计算用户相似度特征...")
# 对于每个用户对，计算他们共同交互的商品数量
user_item_matrix = train_data.groupby(['inviter_id', 'item_id']).size().reset_index(name='count')
user_item_pivot = user_item_matrix.pivot(index='inviter_id', columns='item_id', values='count').fillna(0)

# 为测试集准备特征
print("为测试集准备特征...")
# 合并用户信息
test_with_user = pd.merge(test_data, user_info, left_on='inviter_id', right_on='user_id', how='left')
# 合并商品信息
test_with_item = pd.merge(test_with_user, item_info, on='item_id', how='left')
# 合并用户活跃度
test_with_activity = pd.merge(test_with_item, user_invites, left_on='inviter_id', right_on='user_id', how='left')
test_with_activity = pd.merge(test_with_activity, user_votes, left_on='inviter_id', right_on='user_id', how='left', suffixes=('', '_voter'))
# 合并商品流行度
test_with_popularity = pd.merge(test_with_activity, item_popularity, on='item_id', how='left')

# 用户-商品历史交互特征
print("生成用户-商品历史交互特征...")
# 对于每个测试样本，查找该用户之前是否与该商品有过交互
test_user_item_history = []
for idx, row in tqdm(test_with_popularity.iterrows(), total=len(test_with_popularity)):
    inviter_id = row['inviter_id']
    item_id = row['item_id']
    has_interaction = int(((train_data['inviter_id'] == inviter_id) & (train_data['item_id'] == item_id)).any())
    test_user_item_history.append(has_interaction)

test_with_popularity['has_interaction'] = test_user_item_history

# 生成候选集特征
print("为每个测试样本生成候选集特征...")
# 针对每个测试样本，找出邀请者之前邀请过的回流者
test_candidate_features = []

for idx, row in tqdm(test_with_popularity.iterrows(), total=len(test_with_popularity)):
    inviter_id = row['inviter_id']
    
    # 该邀请者之前邀请过的回流者
    previous_voters = train_data[train_data['inviter_id'] == inviter_id]['voter_id'].value_counts()
    
    if len(previous_voters) > 0:
        # 取前5个最常交互的回流者
        top_voters = previous_voters.nlargest(5).index.tolist()
        # 如果不足5个，用-1填充
        top_voters = top_voters + [-1] * (5 - len(top_voters))
    else:
        # 如果没有历史交互，全部填充-1
        top_voters = [-1] * 5
    
    test_candidate_features.append({
        'triple_id': row['triple_id'],
        'candidate_voter_list': [str(v) for v in top_voters if v != -1]
    })

# 保存候选集特征为JSON文件
with open('results/features/test_candidates.json', 'w', encoding='utf-8') as f:
    for item in test_candidate_features:
        f.write(json.dumps(item) + '\n')

# 保存特征数据
print("保存特征数据...")
user_item_matrix.to_csv('results/features/user_item_matrix.csv', index=False)
user_user_matrix.to_csv('results/features/user_user_matrix.csv', index=False)
user_activity.to_csv('results/features/user_activity.csv', index=False)
item_popularity.to_csv('results/features/item_popularity.csv', index=False)
network_features.to_csv('results/features/network_features.csv', index=False)
user_item_freq.to_csv('results/features/user_item_freq.csv', index=False)
user_cate_freq.to_csv('results/features/user_cate_freq.csv', index=False)
user_cate1_freq.to_csv('results/features/user_cate1_freq.csv', index=False)
test_with_popularity.to_csv('results/features/test_features.csv', index=False)

print("特征工程完成！特征数据保存在 results/features 文件夹中。") 