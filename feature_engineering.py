import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler

def create_user_features(item_share_df, user_info_df):
    """
    根据用户行为和信息创建用户特征
    """
    # 合并用户信息
    user_features = user_info_df.copy()
    
    # 计算用户作为邀请者的统计特征
    inviter_stats = item_share_df.groupby('inviter_id').agg({
        'voter_id': ['count', 'nunique'],  # 邀请的总人数和唯一人数
        'item_id': ['nunique']  # 分享的唯一商品数
    })
    inviter_stats.columns = ['invites_count', 'unique_voters_count', 'unique_items_shared']
    
    # 计算用户作为被邀请者的统计特征
    voter_stats = item_share_df.groupby('voter_id').agg({
        'inviter_id': ['count', 'nunique'],  # 被邀请的总次数和唯一邀请人数
        'item_id': ['nunique']  # 查看的唯一商品数
    })
    voter_stats.columns = ['times_invited', 'unique_inviters_count', 'unique_items_viewed']
    
    # 计算RFM指标（Recency, Frequency, Monetary - 在这里我们主要关注Recency和Frequency）
    # 获取数据集中最大时间作为当前时间参考点
    max_time = item_share_df['timestamp'].max()
    
    # 计算用户最近一次作为邀请者的时间差（Recency）
    inviter_latest = item_share_df.groupby('inviter_id')['timestamp'].max().reset_index()
    inviter_latest['recency_inviter'] = (max_time - inviter_latest['timestamp']).dt.total_seconds() / 86400  # 转换为天数
    inviter_recency = inviter_latest[['inviter_id', 'recency_inviter']].set_index('inviter_id')
    
    # 计算用户最近一次作为被邀请者的时间差（Recency）
    voter_latest = item_share_df.groupby('voter_id')['timestamp'].max().reset_index()
    voter_latest['recency_voter'] = (max_time - voter_latest['timestamp']).dt.total_seconds() / 86400  # 转换为天数
    voter_recency = voter_latest[['voter_id', 'recency_voter']].set_index('voter_id')
    
    # 合并所有用户特征
    user_features = user_features.join(inviter_stats, how='left')
    user_features = user_features.join(voter_stats, how='left')
    user_features = user_features.join(inviter_recency, how='left')
    user_features = user_features.join(voter_recency, how='left')
    
    # 填充缺失值
    user_features.fillna({
        'invites_count': 0,
        'unique_voters_count': 0,
        'unique_items_shared': 0,
        'times_invited': 0,
        'unique_inviters_count': 0,
        'unique_items_viewed': 0,
        'recency_inviter': 999,  # 一个很大的值表示从未作为邀请者
        'recency_voter': 999,  # 一个很大的值表示从未作为被邀请者
    }, inplace=True)
    
    # 计算用户的社交活跃度（邀请次数/被邀请次数的比率）
    user_features['social_activity_ratio'] = user_features['invites_count'] / (user_features['times_invited'] + 1)
    
    # 根据用户的RFM指标对用户进行分类
    user_features['user_activity_score'] = (
        user_features['invites_count'] + 
        user_features['times_invited'] + 
        user_features['unique_items_shared'] + 
        user_features['unique_items_viewed']
    ) / (user_features['recency_inviter'] + user_features['recency_voter'] + 2)  # 加2避免除以0
    
    return user_features

def create_item_features(item_share_df, item_info_df):
    """
    根据商品信息和分享行为创建商品特征
    """
    # 合并商品信息
    item_features = item_info_df.copy()
    
    # 计算商品的统计特征
    item_stats = item_share_df.groupby('item_id').agg({
        'inviter_id': ['count', 'nunique'],  # 商品被分享的总次数和唯一分享者数
        'voter_id': ['nunique']  # 商品的唯一查看者数
    })
    item_stats.columns = ['times_shared', 'unique_sharers', 'unique_viewers']
    
    # 计算商品的流行度（被分享次数）
    item_stats['popularity'] = item_stats['times_shared']
    
    # 计算商品的转化率（唯一查看者/唯一分享者）
    item_stats['conversion_rate'] = item_stats['unique_viewers'] / item_stats['unique_sharers']
    
    # 合并所有商品特征
    item_features = item_features.join(item_stats, how='left')
    
    # 填充缺失值
    item_features.fillna({
        'times_shared': 0,
        'unique_sharers': 0,
        'unique_viewers': 0,
        'popularity': 0,
        'conversion_rate': 0
    }, inplace=True)
    
    return item_features

def create_interaction_features(item_share_df, user_features, item_features):
    """
    创建用户-商品交互特征
    """
    # 创建用户-商品对及其交互次数
    user_item_interactions = item_share_df.groupby(['inviter_id', 'item_id']).size().reset_index(name='interaction_count')
    
    # 合并用户特征和商品特征
    interaction_features = user_item_interactions.copy()
    interaction_features = pd.merge(
        interaction_features, 
        user_features.reset_index(), 
        left_on='inviter_id', 
        right_on='user_id', 
        how='left'
    )
    interaction_features = pd.merge(
        interaction_features, 
        item_features.reset_index(), 
        on='item_id', 
        how='left'
    )
    
    # 创建用户-商品交互特征
    # 例如：用户社交活跃度与商品流行度的结合
    interaction_features['social_popularity'] = interaction_features['social_activity_ratio'] * interaction_features['popularity']
    
    return interaction_features

def create_time_features(item_share_df):
    """
    创建时间相关特征
    """
    time_features = item_share_df.copy()
    
    # 提取时间特征
    time_features['hour'] = time_features['timestamp'].dt.hour
    time_features['day'] = time_features['timestamp'].dt.day
    time_features['month'] = time_features['timestamp'].dt.month
    time_features['dayofweek'] = time_features['timestamp'].dt.dayofweek
    time_features['is_weekend'] = time_features['dayofweek'].isin([5, 6]).astype(int)
    
    # 一天中的时段分类
    time_features['time_period'] = pd.cut(
        time_features['hour'], 
        bins=[0, 6, 12, 18, 24], 
        labels=['night', 'morning', 'afternoon', 'evening']
    )
    
    # 将时段转换为数值型特征
    time_period_encoder = LabelEncoder()
    time_features['time_period_encoded'] = time_period_encoder.fit_transform(time_features['time_period'])
    
    return time_features

def create_graph_features(item_share_df):
    """
    创建图结构相关特征
    """
    import networkx as nx
    
    # 构建用户社交图
    G = nx.DiGraph()
    
    # 添加边：inviter_id -> voter_id
    for _, row in item_share_df.iterrows():
        G.add_edge(row['inviter_id'], row['voter_id'])
    
    # 计算每个用户的中心性指标
    node_degree = dict(G.degree())
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    try:
        # 尝试计算PageRank，但可能因为图太大而内存不足
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
    except:
        # 如果内存不足，则使用近似值
        pagerank = {node: (out_degree.get(node, 0) + 1) / (sum(out_degree.values()) + len(G.nodes())) for node in G.nodes()}
    
    # 将图特征转换为DataFrame
    graph_features = pd.DataFrame({
        'user_id': list(G.nodes()),
        'degree': [node_degree.get(node, 0) for node in G.nodes()],
        'in_degree': [in_degree.get(node, 0) for node in G.nodes()],
        'out_degree': [out_degree.get(node, 0) for node in G.nodes()],
        'pagerank': [pagerank.get(node, 0) for node in G.nodes()]
    })
    
    # 设置索引
    graph_features.set_index('user_id', inplace=True)
    
    return graph_features

def build_features(train_df, user_info_df, item_info_df):
    """
    构建所有特征
    """
    print("创建用户特征...")
    user_features = create_user_features(train_df, user_info_df)
    
    print("创建商品特征...")
    item_features = create_item_features(train_df, item_info_df)
    
    print("创建用户-商品交互特征...")
    interaction_features = create_interaction_features(train_df, user_features, item_features)
    
    print("创建时间特征...")
    time_features = create_time_features(train_df)
    
    print("创建图结构特征...")
    graph_features = create_graph_features(train_df)
    
    # 整合所有特征
    print("整合所有特征...")
    
    # 合并用户特征
    features = pd.merge(
        train_df, 
        user_features.reset_index(),
        left_on='inviter_id', 
        right_on='user_id', 
        how='left',
        suffixes=('', '_inviter')
    )
    
    # 合并被邀请用户特征
    features = pd.merge(
        features, 
        user_features.reset_index(),
        left_on='voter_id', 
        right_on='user_id', 
        how='left',
        suffixes=('', '_voter')
    )
    
    # 合并商品特征
    features = pd.merge(
        features, 
        item_features.reset_index(),
        on='item_id', 
        how='left'
    )
    
    # 合并图结构特征
    features = pd.merge(
        features, 
        graph_features.reset_index(),
        left_on='inviter_id', 
        right_on='user_id', 
        how='left',
        suffixes=('', '_inviter_graph')
    )
    
    features = pd.merge(
        features, 
        graph_features.reset_index(),
        left_on='voter_id', 
        right_on='user_id', 
        how='left',
        suffixes=('', '_voter_graph')
    )
    
    # 提取时间特征并合并
    time_cols = ['hour', 'day', 'month', 'dayofweek', 'is_weekend', 'time_period_encoded']
    features = pd.merge(
        features,
        time_features[['inviter_id', 'item_id', 'timestamp'] + time_cols],
        on=['inviter_id', 'item_id', 'timestamp'],
        how='left'
    )
    
    print(f"最终特征数量: {features.shape[1]}")
    
    return features

if __name__ == "__main__":
    from data_preprocessing import preprocess_data, split_data
    
    # 加载并预处理数据
    item_share_df, user_info_df, item_info_df = preprocess_data()
    
    # 划分训练集和测试集
    train_df, test_df = split_data(item_share_df)
    
    # 构建特征
    train_features = build_features(train_df, user_info_df, item_info_df)
    test_features = build_features(test_df, user_info_df, item_info_df)
    
    # 保存特征数据（可选）
    os.makedirs('processed_data', exist_ok=True)
    train_features.to_csv('processed_data/train_features.csv', index=False)
    test_features.to_csv('processed_data/test_features.csv', index=False) 