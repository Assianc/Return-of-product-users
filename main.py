import os
import argparse
import pandas as pd
import numpy as np
import time
import sys
import psutil
from datetime import datetime
from alive_progress import alive_bar
import logging
import json
from tqdm import tqdm

# 导入自定义模块
from data_preprocessing import preprocess_data, split_data
from feature_engineering import build_features

# 深度学习和优化库
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# GNN相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import xgboost as xgb
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, add_self_loops

# PyTorch Geometric导入
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence

# 添加内存监控函数
def get_memory_usage():
    """获取当前进程内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    return memory_usage_mb

# 打印内存使用情况
def print_memory_usage(prefix=''):
    memory_usage = get_memory_usage()
    print(f"{prefix}当前内存占用: {memory_usage:.2f} MB")

# 自定义进度条函数
def simple_progress_bar(iteration, total, prefix='', length=50, fill='█', print_end='\r'):
    """
    简单的进度条函数，不依赖第三方库
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% ({iteration}/{total})')
    sys.stdout.flush()
    if iteration == total:
        print()

# 创建数据序列生成器，用于LSTM输入
def create_sequence_data(X, y, sequence_length):
    """
    将特征数据转换为序列形式，适用于LSTM模型
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])
    return np.array(X_seq), np.array(y_seq)

# 构建CNN+LSTM混合模型
class CNNLSTMModel:
    def __init__(self, input_shape, num_classes=1, sequence_length=5, lstm_units=50, 
                 filters=64, kernel_size=3, dense_units=128, 
                 dropout_rate=0.3, learning_rate=0.001, random_state=42, is_classification=True):
        """
        初始化CNN+LSTM混合模型
        
        参数:
            input_shape: 输入特征的形状 (特征维度)
            num_classes: 分类任务的类别数量，为1则是二分类/回归
            sequence_length: LSTM序列长度
            lstm_units: LSTM单元数量
            filters: CNN过滤器数量
            kernel_size: CNN卷积核大小
            dense_units: 全连接层单元数量
            dropout_rate: Dropout比率
            learning_rate: 学习率
            random_state: 随机种子
            is_classification: 是否为分类任务
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.filters = filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.is_classification = is_classification
        self.model = None
        
        # 设置随机种子
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def build_model(self):
        """构建CNN+LSTM混合模型"""
        # 输入层，形状为(sequence_length, features)
        inputs = Input(shape=(self.sequence_length, self.input_shape))
        
        # CNN部分
        cnn = Conv1D(filters=self.filters, kernel_size=self.kernel_size, 
                     activation='relu')(inputs)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Dropout(self.dropout_rate)(cnn)
        
        # LSTM部分
        lstm_out = LSTM(self.lstm_units, return_sequences=False)(cnn)
        lstm_out = Dropout(self.dropout_rate)(lstm_out)
        
        # 全连接层
        dense1 = Dense(self.dense_units, activation='relu')(lstm_out)
        dense1 = Dropout(self.dropout_rate)(dense1)
        
        # 输出层 - 根据任务类型调整
        if self.is_classification:
            if self.num_classes == 1:  # 二分类
                output = Dense(1, activation='sigmoid')(dense1)
                loss_function = 'binary_crossentropy'
                metrics = ['accuracy']
            else:  # 多分类
                output = Dense(self.num_classes, activation='softmax')(dense1)
                loss_function = 'categorical_crossentropy'
                metrics = ['accuracy']
        else:  # 回归
            output = Dense(1, activation='linear')(dense1)
            loss_function = 'mse'
            metrics = ['mae']
        
        # 编译模型
        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss_function,
            metrics=metrics
        )
        
        self.model = model
        return model
    
    def fit(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, verbose=1):
        """
        训练模型
        """
        # 确保数据为序列形式
        if len(X_train.shape) != 3:
            print("警告：输入数据不是序列形式，尝试转换...")
            X_train, y_train = create_sequence_data(X_train, y_train, self.sequence_length)
            X_val, y_val = create_sequence_data(X_val, y_val, self.sequence_length)
        
        # 创建早停和检查点回调
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('best_cnn_lstm_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X):
        """
        使用模型进行预测
        """
        # 确保数据为序列形式
        if len(X.shape) != 3:
            print("警告：输入数据不是序列形式，尝试转换...")
            sequence_X, _ = create_sequence_data(X, np.zeros(len(X)), self.sequence_length)
            return self.model.predict(sequence_X)
        
        return self.model.predict(X)
    
    def save(self, path):
        """保存模型"""
        self.model.save(path)
    
    def load(self, path):
        """加载模型"""
        self.model = load_model(path)
        return self.model

# 贝叶斯优化CNN+LSTM模型超参数
def bayesian_optimize_cnn_lstm(X_train, y_train, X_val, y_val, n_iter=20, random_state=42):
    """
    使用贝叶斯优化方法寻找CNN+LSTM模型的最佳超参数
    
    参数:
        X_train: 训练特征数据
        y_train: 训练标签数据
        X_val: 验证特征数据
        y_val: 验证标签数据
        n_iter: 贝叶斯优化迭代次数
        random_state: 随机种子
    
    返回:
        最佳超参数
    """
    print("开始贝叶斯优化CNN+LSTM模型超参数...")
    
    # 确保数据是序列形式
    sequence_length = 5  # 初始序列长度
    if len(X_train.shape) != 3:
        print("将数据转换为序列形式...")
        X_train_seq, y_train_seq = create_sequence_data(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequence_data(X_val, y_val, sequence_length)
    else:
        X_train_seq, y_train_seq = X_train, y_train
        X_val_seq, y_val_seq = X_val, y_val
    
    # 定义超参数搜索空间
    param_space = {
        'sequence_length': Integer(3, 10),
        'lstm_units': Integer(32, 256),
        'filters': Integer(32, 128),
        'kernel_size': Integer(2, 5),
        'dense_units': Integer(64, 256),
        'dropout_rate': Real(0.1, 0.5),
        'learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
        'batch_size': Integer(16, 128)
    }
    
    # 定义评估函数
    def evaluate_model(params):
        # 提取参数
        seq_len = params['sequence_length']
        lstm_units = params['lstm_units']
        filters = params['filters']
        kernel_size = params['kernel_size']
        dense_units = params['dense_units']
        dropout_rate = params['dropout_rate']
        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        
        # 如果序列长度变化，需要重新生成序列数据
        if seq_len != sequence_length:
            X_train_seq_new, y_train_seq_new = create_sequence_data(X_train, y_train, seq_len)
            X_val_seq_new, y_val_seq_new = create_sequence_data(X_val, y_val, seq_len)
        else:
            X_train_seq_new, y_train_seq_new = X_train_seq, y_train_seq
            X_val_seq_new, y_val_seq_new = X_val_seq, y_val_seq
        
        # 构建模型
        model = CNNLSTMModel(
            input_shape=X_train.shape[1],
            num_classes=1,
            sequence_length=seq_len,
            lstm_units=lstm_units,
            filters=filters,
            kernel_size=kernel_size,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            random_state=random_state,
            is_classification=True
        )
        
        # 构建并训练模型
        model.build_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        history = model.model.fit(
            X_train_seq_new, y_train_seq_new,
            validation_data=(X_val_seq_new, y_val_seq_new),
            epochs=20,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # 返回验证集上的损失
        val_loss = min(history.history['val_loss'])
        return val_loss
    
    # 运行贝叶斯优化
    optimizer = BayesSearchCV(
        estimator=None,
        search_spaces=param_space,
        n_iter=n_iter,
        scoring=None,  # 使用自定义评估函数
        cv=None,
        n_jobs=1,
        refit=False,
        random_state=random_state
    )
    
    # 手动执行贝叶斯优化
    best_params = None
    best_score = float('inf')
    
    print(f"开始执行{n_iter}次贝叶斯优化迭代...")
    with alive_bar(n_iter, title='贝叶斯优化进度') as bar:
        for i in range(n_iter):
            # 从搜索空间中采样参数
            if i == 0:
                # 第一次迭代使用默认参数
                params = {
                    'sequence_length': 5,
                    'lstm_units': 64,
                    'filters': 64,
                    'kernel_size': 3,
                    'dense_units': 128,
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001,
                    'batch_size': 32
                }
            else:
                params = optimizer.ask()
            
            # 评估参数
            score = evaluate_model(params)
            
            # 更新优化器
            optimizer.tell(params, score)
            
            # 更新最佳参数
            if score < best_score:
                best_score = score
                best_params = params
                print(f"\n迭代 {i+1}/{n_iter} - 发现新的最佳参数: {best_params}")
                print(f"最佳验证损失: {best_score:.4f}")
            
            # 更新进度条
            bar()
    
    print(f"贝叶斯优化完成!")
    print(f"最佳超参数: {best_params}")
    print(f"最佳验证损失: {best_score:.4f}")
    
    return best_params

# GNN+XGBoost混合模型
class HybridModel:
    def __init__(self, gnn_model, link_predictor, node_features, edge_index, node_mapping, use_xgboost=True):
        """
        集成GNN和XGBoost的混合模型 - 简化版，适用于小数据集
        
        参数:
            gnn_model: 训练好的GNN模型
            link_predictor: 训练好的链接预测器
            node_features: 节点特征
            edge_index: 图的边
            node_mapping: 节点ID到索引的映射
            use_xgboost: 是否使用XGBoost
        """
        self.gnn_model = gnn_model
        self.link_predictor = link_predictor
        self.node_features = node_features
        self.edge_index = edge_index
        self.node_mapping = node_mapping
        self.use_xgboost = use_xgboost
        self.xgb_model = None
        
    def train_xgboost(self, train_data, val_data, params=None):
        """
        训练XGBoost模型
        
        参数:
            train_data: 训练数据
            val_data: 验证数据
            params: XGBoost参数
        """
        if not self.use_xgboost:
            print("XGBoost已禁用，跳过训练")
            return
        
        print("训练XGBoost模型...")
        
        # 准备XGBoost特征 - 使用节点嵌入
        X_train = self._prepare_xgb_features(train_data)
        y_train = train_data['labels']
        
        # 验证数据
        X_val = self._prepare_xgb_features(val_data)
        y_val = val_data['labels']
        
        # 默认参数 - 简化配置适合小数据集
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 4,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'num_boost_round': 50  # 减少迭代次数
            }
        
        # 创建DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 训练
        evals = [(dtrain, 'train'), (dval, 'validation')]
        self.xgb_model = xgb.train(params, dtrain, num_boost_round=params['num_boost_round'], 
                                   evals=evals, early_stopping_rounds=5, verbose_eval=10)
        
        print("XGBoost模型训练完成")
    
    def _prepare_xgb_features(self, data):
        """准备XGBoost特征"""
        # 获取源节点和目标节点
        src_nodes = data['src_nodes']
        dst_nodes = data['dst_nodes']
        
        # 使用GNN获取节点嵌入
        self.gnn_model.eval()
        with torch.no_grad():
            node_embeddings = self.gnn_model(self.node_features, self.edge_index)
            
        # 拼接源节点和目标节点的特征
        features = []
        for i in range(len(src_nodes)):
            src_emb = node_embeddings[src_nodes[i]].cpu().numpy()
            dst_emb = node_embeddings[dst_nodes[i]].cpu().numpy()
            
            # 拼接特征 - 简化为嵌入特征和节点类型
            feature = np.concatenate([src_emb, dst_emb])
            features.append(feature)
        
        return np.array(features)
    
    def predict(self, inviter_id, item_id, candidates, timestamp=None):
        """
        推荐候选voter
        
        参数:
            inviter_id: 邀请者ID
            item_id: 物品ID
            candidates: 候选voter ID列表
            timestamp: 时间戳
        
        返回:
            推荐的top-5 voter ID列表
        """
        try:
            # 转换为节点索引
            inviter_node = f"u_{inviter_id}"
            item_node = f"i_{item_id}"
            
            # 检查节点是否在图中
            if inviter_node not in self.node_mapping or item_node not in self.node_mapping:
                print(f"Warning: {inviter_node} 或 {item_node} 不在图中，随机推荐")
                # 随机返回5个候选
                import random
                return random.sample(candidates, min(5, len(candidates)))
            
            inviter_idx = self.node_mapping[inviter_node]
            item_idx = self.node_mapping[item_node]
            
            # 处理候选voter
            valid_candidates = []
            valid_indices = []
            for voter_id in candidates:
                voter_node = f"u_{voter_id}"
                if voter_node in self.node_mapping:
                    valid_candidates.append(voter_id)
                    valid_indices.append(self.node_mapping[voter_node])
            
            # 如果有效候选太少，随机返回
            if len(valid_candidates) < 5:
                print(f"Warning: 只有 {len(valid_candidates)} 个有效候选voter")
                if len(valid_candidates) == 0:
                    import random
                    return random.sample(candidates, min(5, len(candidates)))
            
            # 使用GNN+XGBoost预测
            self.gnn_model.eval()
            self.link_predictor.eval()
            
            with torch.no_grad():
                # 获取节点嵌入
                node_embeddings = self.gnn_model(self.node_features, self.edge_index)
                
                # 获取item嵌入
                item_emb = node_embeddings[item_idx]
                
                # 预测item->voter链接
                scores = []
                for i, voter_idx in enumerate(valid_indices):
                    try:
                        voter_emb = node_embeddings[voter_idx]
                        # GNN预测分数
                        gnn_score = self.link_predictor(item_emb.unsqueeze(0), voter_emb.unsqueeze(0)).item()
                        
                        # XGBoost预测分数
                        xgb_score = 0
                        if self.use_xgboost and self.xgb_model is not None:
                            # 准备XGBoost特征
                            feature = np.concatenate([item_emb.cpu().numpy(), voter_emb.cpu().numpy()])
                            dmat = xgb.DMatrix(feature.reshape(1, -1))
                            xgb_score = self.xgb_model.predict(dmat)[0]
                        
                        # 结合分数 - 简单加权平均
                        final_score = 0.7 * gnn_score + 0.3 * xgb_score if self.use_xgboost else gnn_score
                        scores.append((valid_candidates[i], final_score))
                    except Exception as e:
                        # 如果单个预测失败，跳过这个候选人
                        print(f"预测候选人 {valid_candidates[i]} 时出错: {e}")
                        continue
                
                # 如果没有得到任何有效分数，随机返回
                if len(scores) == 0:
                    print("未得到任何有效预测分数，使用随机推荐")
                    import random
                    return random.sample(candidates, min(5, len(candidates)))
                
                # 按分数降序排序
                scores.sort(key=lambda x: x[1], reverse=True)
                
                # 确保有足够的推荐
                recommended = [voter_id for voter_id, _ in scores[:5]]
                if len(recommended) < 5:
                    # 随机添加一些候选人以填满5个位置
                    import random
                    remaining = [c for c in candidates if c not in recommended]
                    additional = random.sample(remaining, min(5 - len(recommended), len(remaining)))
                    recommended.extend(additional)
                
                # 返回top-5
                return recommended[:5]
                
        except Exception as e:
            # 捕获所有异常，确保始终返回5个推荐
            print(f"预测过程中出现错误: {e}")
            import random
            return random.sample(candidates, min(5, len(candidates)))

# 图数据处理相关函数
def build_graph_from_interactions(df, user_info_df, item_info_df):
    """
    构建用户-物品二分图
    
    参数:
        df: 交互数据，包含inviter_id, item_id, voter_id, timestamp
        user_info_df: 用户信息
        item_info_df: 物品信息
    
    返回:
        G: NetworkX图对象
        node_mapping: 节点ID到索引的映射
        features: 节点特征矩阵
    """
    print("构建交互图...")
    
    # 创建一个空的无向图
    G = nx.Graph()
    
    # 将用户(inviter和voter)和物品添加为节点
    inviters = set(df['inviter_id'].unique())
    voters = set(df['voter_id'].unique())
    items = set(df['item_id'].unique())
    
    # 所有用户
    all_users = inviters.union(voters)
    print(f"图中的用户数量: {len(all_users)}, 物品数量: {len(items)}")
    
    # 为节点添加类型前缀，以区分用户和物品节点
    user_nodes = [f"u_{user}" for user in all_users]
    item_nodes = [f"i_{item}" for item in items]
    
    # 添加节点
    G.add_nodes_from(user_nodes, node_type='user')
    G.add_nodes_from(item_nodes, node_type='item')
    
    # 创建节点ID到索引的映射
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    
    # 添加边 (inviter -> item) 和 (item -> voter)
    edges = []
    edge_timestamps = []
    
    # 按时间戳排序，确保时间顺序
    df = df.sort_values('timestamp')
    
    # 添加交互边
    print("添加交互边...")
    with tqdm(total=len(df), desc="处理交互数据") as pbar:
        for _, row in df.iterrows():
            inviter = f"u_{row['inviter_id']}"
            item = f"i_{row['item_id']}"
            voter = f"u_{row['voter_id']}"
            timestamp = pd.to_datetime(row['timestamp']).timestamp()
            
            # 添加边
            edges.append((inviter, item))
            edges.append((item, voter))
            
            # 边的时间戳
            edge_timestamps.append(timestamp)
            edge_timestamps.append(timestamp)
            
            pbar.update(1)
    
    # 将边添加到图中
    print("添加边到图中...")
    G.add_edges_from(edges)
    
    # 为边添加时间戳属性
    print("添加时间戳属性...")
    with tqdm(total=len(edges), desc="添加边属性") as pbar:
        for (u, v), ts in zip(edges, edge_timestamps):
            G[u][v]['timestamp'] = ts
            pbar.update(1)
    
    # 为节点添加特征
    # 1. 用户特征
    print("添加用户特征...")
    with tqdm(total=len(all_users), desc="添加用户特征") as pbar:
        for user in all_users:
            user_node = f"u_{user}"
            if user in user_info_df['user_id'].values:
                user_data = user_info_df[user_info_df['user_id'] == user].iloc[0]
                G.nodes[user_node]['gender'] = user_data['user_gender']
                G.nodes[user_node]['age'] = user_data['user_age']
                G.nodes[user_node]['level'] = user_data['user_level']
            else:
                # 对于未知用户，使用默认值
                G.nodes[user_node]['gender'] = -1
                G.nodes[user_node]['age'] = -1
                G.nodes[user_node]['level'] = -1
            pbar.update(1)
    
    # 2. 物品特征
    print("添加物品特征...")
    with tqdm(total=len(items), desc="添加物品特征") as pbar:
        for item in items:
            item_node = f"i_{item}"
            if item in item_info_df['item_id'].values:
                item_data = item_info_df[item_info_df['item_id'] == item].iloc[0]
                G.nodes[item_node]['cate_id'] = item_data['cate_id']
                G.nodes[item_node]['cate_level1_id'] = item_data['cate_level1_id']
                G.nodes[item_node]['brand_id'] = item_data['brand_id']
                G.nodes[item_node]['shop_id'] = item_data['shop_id']
            else:
                # 对于未知物品，使用默认值
                G.nodes[item_node]['cate_id'] = -1
                G.nodes[item_node]['cate_level1_id'] = -1
                G.nodes[item_node]['brand_id'] = -1
                G.nodes[item_node]['shop_id'] = -1
            pbar.update(1)
    
    print(f"图构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    
    # 准备节点特征矩阵
    features = prepare_node_features(G, node_mapping, user_info_df, item_info_df)
    
    return G, node_mapping, features

def prepare_node_features(G, node_mapping, user_info_df, item_info_df):
    """
    准备节点特征矩阵 - 简化版，适用于小数据集
    
    参数:
        G: NetworkX图对象
        node_mapping: 节点ID到索引的映射
        user_info_df: 用户信息
        item_info_df: 物品信息
    
    返回:
        features: 节点特征矩阵
    """
    print("准备节点特征...")
    
    # 初始化特征矩阵
    num_nodes = len(node_mapping)
    # 特征维度: 节点类型(1) + 基本用户特征(3) + 基本物品特征(4) + 拓扑特征(3)
    # 为小数据集减少特征维度
    feature_dim = 11
    features = np.zeros((num_nodes, feature_dim))
    
    # 为每个节点创建特征
    with tqdm(total=num_nodes, desc="创建节点特征") as pbar:
        for node, idx in node_mapping.items():
            node_type = G.nodes[node]['node_type']
            
            # 节点类型特征 (one-hot形式)
            features[idx, 0] = 1 if node_type == 'user' else 0
            
            # 节点特征
            if node_type == 'user':
                # 用户特征: 性别, 年龄, 等级
                # 可以根据实际情况添加更多特征
                features[idx, 1] = G.nodes[node].get('gender', 0)
                features[idx, 2] = G.nodes[node].get('age', 0)
                features[idx, 3] = G.nodes[node].get('level', 0)
            else:  # item
                # 物品特征: 类目ID, 一级类目ID, 品牌ID, 店铺ID
                features[idx, 4] = G.nodes[node].get('cate_id', 0)
                features[idx, 5] = G.nodes[node].get('cate_level1_id', 0)
                features[idx, 6] = G.nodes[node].get('brand_id', 0)
                features[idx, 7] = G.nodes[node].get('shop_id', 0)
            
            # 为所有节点添加基本拓扑特征
            # 度
            features[idx, 8] = G.degree(node)
            # 聚类系数 - 简化版本中跳过复杂计算
            features[idx, 9] = 0
            # 中心性 - 简化版本中使用度作为近似
            features[idx, 10] = G.degree(node) / (len(G) - 1) if len(G) > 1 else 0
            
            pbar.update(1)
    
    # 简单标准化
    print("正在标准化特征...")
    for i in range(feature_dim):
        if np.max(features[:, i]) > 0:
            features[:, i] = features[:, i] / (np.max(features[:, i]) + 1e-6)
    
    print(f"节点特征准备完成: {features.shape}")
    
    return features

def prepare_link_prediction_data(G, node_mapping, df_train, df_val, negative_samples=5):
    """
    准备链接预测的训练和验证数据
    
    参数:
        G: NetworkX图
        node_mapping: 节点到索引的映射
        df_train: 训练集DataFrame，包含inviter_id, item_id, voter_id
        df_val: 验证集DataFrame
        negative_samples: 每个正样本对应的负样本数量
    
    返回:
        train_data: 训练数据 (src_nodes, dst_nodes, labels)
        val_data: 验证数据
    """
    print("准备链接预测数据...")
    
    # 准备训练数据
    src_nodes_train = []
    dst_nodes_train = []
    labels_train = []
    timestamps_train = []
    
    # 正样本：实际存在的inviter->item和item->voter边
    print("处理训练集正样本...")
    with tqdm(total=len(df_train), desc="处理训练集正样本") as pbar:
        for _, row in df_train.iterrows():
            inviter = f"u_{row['inviter_id']}"
            item = f"i_{row['item_id']}"
            voter = f"u_{row['voter_id']}"
            
            # 转换为索引
            inviter_idx = node_mapping[inviter]
            item_idx = node_mapping[item]
            voter_idx = node_mapping[voter]
            
            timestamp = pd.to_datetime(row['timestamp']).timestamp()
            
            # inviter -> item 边
            src_nodes_train.append(inviter_idx)
            dst_nodes_train.append(item_idx)
            labels_train.append(1)  # 正样本
            timestamps_train.append(timestamp)
            
            # item -> voter 边
            src_nodes_train.append(item_idx)
            dst_nodes_train.append(voter_idx)
            labels_train.append(1)  # 正样本
            timestamps_train.append(timestamp)
            
            pbar.update(1)
    
    # 负样本：随机生成不存在的边
    all_users = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'user']
    all_items = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'item']
    
    all_user_indices = [node_mapping[n] for n in all_users]
    all_item_indices = [node_mapping[n] for n in all_items]
    
    # 为每个正样本生成负样本
    print("生成训练集负样本...")
    total_negative_samples = len(df_train) * negative_samples
    with tqdm(total=total_negative_samples, desc="生成训练集负样本") as pbar:
        for i in range(0, len(src_nodes_train), 2):  # 每两个边对应一个交互
            inviter_idx = src_nodes_train[i]
            item_idx = dst_nodes_train[i]
            voter_idx = dst_nodes_train[i+1]
            timestamp = timestamps_train[i]
            
            # 为inviter->item生成负样本 (inviter->其他item)
            for _ in range(negative_samples // 2):
                neg_item_idx = np.random.choice(all_item_indices)
                # 确保这个边不存在
                inviter_node = list(node_mapping.keys())[list(node_mapping.values()).index(inviter_idx)]
                neg_item_node = list(node_mapping.keys())[list(node_mapping.values()).index(neg_item_idx)]
                if not G.has_edge(inviter_node, neg_item_node):
                    src_nodes_train.append(inviter_idx)
                    dst_nodes_train.append(neg_item_idx)
                    labels_train.append(0)  # 负样本
                    timestamps_train.append(timestamp)
                
                pbar.update(1)
            
            # 为item->voter生成负样本 (item->其他user)
            for _ in range(negative_samples // 2):
                neg_voter_idx = np.random.choice(all_user_indices)
                # 确保这个边不存在
                item_node = list(node_mapping.keys())[list(node_mapping.values()).index(item_idx)]
                neg_voter_node = list(node_mapping.keys())[list(node_mapping.values()).index(neg_voter_idx)]
                if not G.has_edge(item_node, neg_voter_node):
                    src_nodes_train.append(item_idx)
                    dst_nodes_train.append(neg_voter_idx)
                    labels_train.append(0)  # 负样本
                    timestamps_train.append(timestamp)
                
                pbar.update(1)
    
    # 同样处理验证集
    src_nodes_val = []
    dst_nodes_val = []
    labels_val = []
    timestamps_val = []
    
    print("处理验证集正样本...")
    with tqdm(total=len(df_val), desc="处理验证集正样本") as pbar:
        for _, row in df_val.iterrows():
            inviter = f"u_{row['inviter_id']}"
            item = f"i_{row['item_id']}"
            voter = f"u_{row['voter_id']}"
            
            # 转换为索引
            inviter_idx = node_mapping[inviter]
            item_idx = node_mapping[item]
            voter_idx = node_mapping[voter]
            
            timestamp = pd.to_datetime(row['timestamp']).timestamp()
            
            # inviter -> item 边
            src_nodes_val.append(inviter_idx)
            dst_nodes_val.append(item_idx)
            labels_val.append(1)
            timestamps_val.append(timestamp)
            
            # item -> voter 边
            src_nodes_val.append(item_idx)
            dst_nodes_val.append(voter_idx)
            labels_val.append(1)
            timestamps_val.append(timestamp)
            
            pbar.update(1)
    
    # 同样为验证集生成负样本
    print("生成验证集负样本...")
    total_negative_samples_val = len(df_val) * negative_samples
    with tqdm(total=total_negative_samples_val, desc="生成验证集负样本") as pbar:
        for i in range(0, len(src_nodes_val), 2):
            inviter_idx = src_nodes_val[i]
            item_idx = dst_nodes_val[i]
            voter_idx = dst_nodes_val[i+1]
            timestamp = timestamps_val[i]
            
            # 为inviter->item生成负样本
            for _ in range(negative_samples // 2):
                neg_item_idx = np.random.choice(all_item_indices)
                inviter_node = list(node_mapping.keys())[list(node_mapping.values()).index(inviter_idx)]
                neg_item_node = list(node_mapping.keys())[list(node_mapping.values()).index(neg_item_idx)]
                if not G.has_edge(inviter_node, neg_item_node):
                    src_nodes_val.append(inviter_idx)
                    dst_nodes_val.append(neg_item_idx)
                    labels_val.append(0)
                    timestamps_val.append(timestamp)
                
                pbar.update(1)
            
            # 为item->voter生成负样本
            for _ in range(negative_samples // 2):
                neg_voter_idx = np.random.choice(all_user_indices)
                item_node = list(node_mapping.keys())[list(node_mapping.values()).index(item_idx)]
                neg_voter_node = list(node_mapping.keys())[list(node_mapping.values()).index(neg_voter_idx)]
                if not G.has_edge(item_node, neg_voter_node):
                    src_nodes_val.append(item_idx)
                    dst_nodes_val.append(neg_voter_idx)
                    labels_val.append(0)
                    timestamps_val.append(timestamp)
                
                pbar.update(1)
    
    # 转换为numpy数组
    print("转换为numpy数组...")
    train_data = {
        'src_nodes': np.array(src_nodes_train),
        'dst_nodes': np.array(dst_nodes_train),
        'labels': np.array(labels_train),
        'timestamps': np.array(timestamps_train)
    }
    
    val_data = {
        'src_nodes': np.array(src_nodes_val),
        'dst_nodes': np.array(dst_nodes_val),
        'labels': np.array(labels_val),
        'timestamps': np.array(timestamps_val)
    }
    
    print(f"链接预测数据准备完成: {len(train_data['labels'])} 训练样本, {len(val_data['labels'])} 验证样本")
    
    return train_data, val_data

# GNN模型定义
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2, gnn_type='sage'):
        """
        图神经网络模型 - 简化版，适用于小数据集
        
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出特征维度
            dropout: Dropout比率
            gnn_type: GNN类型 ('gcn', 'sage', 'gat')
        """
        super(GNN, self).__init__()
        
        self.dropout = dropout
        
        # 选择GNN层类型
        if gnn_type == 'gcn':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif gnn_type == 'sage':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif gnn_type == 'gat':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, out_channels)
        else:
            raise ValueError(f"不支持的GNN类型: {gnn_type}")
    
    def forward(self, x, edge_index):
        # 第一层GNN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层GNN
        x = self.conv2(x, edge_index)
        
        return x

# 链接预测模型
class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0.2):
        """
        链接预测模型 - 简化版，适用于小数据集
        
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出维度 (通常为1，表示链接存在的概率)
            dropout: Dropout比率
        """
        super(LinkPredictor, self).__init__()
        
        # 简化为单层MLP
        self.lin1 = nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_src, x_dst):
        """
        前向传播
        
        参数:
            x_src: 源节点特征
            x_dst: 目标节点特征
        """
        # 拼接源节点和目标节点的特征
        x = torch.cat([x_src, x_dst], dim=1)
        
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return torch.sigmoid(x)  # 对于二分类，使用sigmoid激活

def main(args):
    start_time = time.time()
    
    # 创建必要的目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'processed_data'), exist_ok=True)
    
    # 记录日志
    log_file = os.path.join(args.output_dir, 'experiment_log.txt')
    
    def log(message):
        """记录日志信息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'[{timestamp}] {message}\n')
        print(message)
    
    log(f"开始执行实验，参数配置：{args}")
    
    # 数据预处理
    log("开始数据预处理...")
    try:
        print("正在加载和预处理数据...")
        item_share_df, user_info_df, item_info_df = preprocess_data()
        
        # 使用完整数据集进行训练
        original_size = len(item_share_df)
        log(f"使用完整数据集: {original_size} 条记录")
        
        # 保存预处理后的数据
        processed_data_dir = os.path.join(args.output_dir, 'processed_data')
        print("保存预处理后的数据...")
        user_info_df.to_csv(os.path.join(processed_data_dir, 'user_info.csv'), index=False)
        item_info_df.to_csv(os.path.join(processed_data_dir, 'item_info.csv'), index=False)
        item_share_df.to_csv(os.path.join(processed_data_dir, 'item_share.csv'), index=False)
        
        log(f"数据预处理完成，商品分享记录数量: {len(item_share_df)}, 用户数量: {len(user_info_df)}, 商品数量: {len(item_info_df)}")
    except Exception as e:
        log(f"数据预处理失败: {str(e)}")
        return
    
    # 划分训练集和测试集
    log("划分数据集...")
    try:
        print("正在划分训练集和测试集...")
        train_df, test_df = split_data(item_share_df, test_size=args.test_size)
        
        # 保存划分后的数据
        train_df.to_csv(os.path.join(processed_data_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(processed_data_dir, 'test.csv'), index=False)
        
        log(f"数据集划分完成，训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    except Exception as e:
        log(f"数据集划分失败: {str(e)}")
        return
    
    # 构建图
    log("构建社交图谱...")
    try:
        # 构建图
        G, node_mapping, node_features = build_graph_from_interactions(train_df, user_info_df, item_info_df)
        
        # 转换为PyTorch Geometric数据格式
        edge_index = []
        for u, v in G.edges():
            edge_index.append([node_mapping[u], node_mapping[v]])
            # 添加反向边
            edge_index.append([node_mapping[v], node_mapping[u]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # 保存图数据
        torch.save(edge_index, os.path.join(processed_data_dir, 'edge_index.pt'))
        torch.save(node_features, os.path.join(processed_data_dir, 'node_features.pt'))
        import pickle
        with open(os.path.join(processed_data_dir, 'node_mapping.pkl'), 'wb') as f:
            pickle.dump(node_mapping, f)
        
        log(f"图构建完成，节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    except Exception as e:
        log(f"图构建失败: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return
    
    # 准备链接预测数据
    log("准备链接预测数据...")
    try:
        # 划分训练集用于验证
        train_df, val_df = train_test_split(train_df, test_size=args.val_size, random_state=args.random_state)
        
        # 准备训练和验证数据
        train_data, val_data = prepare_link_prediction_data(G, node_mapping, train_df, val_df)
        
        log(f"链接预测数据准备完成，训练样本: {len(train_data['labels'])}, 验证样本: {len(val_data['labels'])}")
    except Exception as e:
        log(f"链接预测数据准备失败: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return
    
    # 训练GNN模型
    log("训练GNN模型...")
    try:
        # 模型参数 - 为大数据集调整
        in_channels = node_features.shape[1]
        hidden_channels = args.hidden_channels  # 使用命令行参数
        out_channels = args.embedding_dim      # 使用命令行参数
        
        # 创建GNN模型
        gnn_model = GNN(in_channels, hidden_channels, out_channels, 
                        dropout=args.dropout, gnn_type=args.gnn_type)
        
        # 创建链接预测器
        link_predictor = LinkPredictor(out_channels, hidden_channels)
        
        # 转换为PyTorch张量
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # 优化器
        optimizer = torch.optim.Adam([
            {'params': gnn_model.parameters()},
            {'params': link_predictor.parameters()}
        ], lr=args.learning_rate)
        
        # 损失函数
        criterion = nn.BCELoss()
        
        # 训练数据
        train_src = torch.tensor(train_data['src_nodes'], dtype=torch.long)
        train_dst = torch.tensor(train_data['dst_nodes'], dtype=torch.long)
        train_labels = torch.tensor(train_data['labels'], dtype=torch.float)
        
        # 验证数据
        val_src = torch.tensor(val_data['src_nodes'], dtype=torch.long)
        val_dst = torch.tensor(val_data['dst_nodes'], dtype=torch.long)
        val_labels = torch.tensor(val_data['labels'], dtype=torch.float)
        
        # 对于大数据集，减少训练轮数
        actual_epochs = min(30, args.epochs)
        log(f"使用训练轮数: {actual_epochs} (原计划: {args.epochs})")
        
        # 训练循环
        best_val_loss = float('inf')
        patience = 3  # 减少早停的耐心值
        counter = 0
        
        for epoch in range(actual_epochs):
            # 训练
            gnn_model.train()
            link_predictor.train()
            
            # 获取节点嵌入
            node_embeddings = gnn_model(node_features, edge_index)
            
            # 正向传播
            src_embeddings = node_embeddings[train_src]
            dst_embeddings = node_embeddings[train_dst]
            
            pred = link_predictor(src_embeddings, dst_embeddings).squeeze()
            loss = criterion(pred, train_labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 验证
            gnn_model.eval()
            link_predictor.eval()
            
            with torch.no_grad():
                node_embeddings = gnn_model(node_features, edge_index)
                
                src_embeddings = node_embeddings[val_src]
                dst_embeddings = node_embeddings[val_dst]
                
                pred = link_predictor(src_embeddings, dst_embeddings).squeeze()
                val_loss = criterion(pred, val_labels)
                
                # 计算准确率
                pred_binary = (pred > 0.5).float()
                val_acc = (pred_binary == val_labels).sum().item() / len(val_labels)
            
            # 打印进度
            if (epoch + 1) % 1 == 0:  # 每个epoch都打印
                log(f"Epoch {epoch+1}/{actual_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                
                # 保存最佳模型
                torch.save(gnn_model.state_dict(), os.path.join(args.output_dir, 'models', 'gnn_model.pt'))
                torch.save(link_predictor.state_dict(), os.path.join(args.output_dir, 'models', 'link_predictor.pt'))
            else:
                counter += 1
                if counter >= patience:
                    log(f"早停: {patience} 个epoch内验证损失没有改善")
                    break
        
        # 加载最佳模型
        gnn_model.load_state_dict(torch.load(os.path.join(args.output_dir, 'models', 'gnn_model.pt')))
        link_predictor.load_state_dict(torch.load(os.path.join(args.output_dir, 'models', 'link_predictor.pt')))
        
        log("GNN模型训练完成")
    except Exception as e:
        log(f"GNN模型训练失败: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return
    
    # 训练XGBoost模型
    log("训练XGBoost模型...")
    try:
        # 创建混合模型
        hybrid_model = HybridModel(gnn_model, link_predictor, node_features, edge_index, node_mapping, 
                                  use_xgboost=args.use_xgboost)
        
        # 训练XGBoost - 对于完整数据集，使用更合适的参数
        if args.use_xgboost:
            # 为完整数据集配置XGBoost参数
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,  # 适当增加深度
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'num_boost_round': 100  # 增加迭代次数
            }
            hybrid_model.train_xgboost(train_data, val_data, params=xgb_params)
            
            # 保存XGBoost模型
            if hybrid_model.xgb_model is not None:
                hybrid_model.xgb_model.save_model(os.path.join(args.output_dir, 'models', 'xgboost_model.json'))
                log("XGBoost模型已保存")
        
        log("混合模型训练完成")
        
        # 获取所有用户ID
        all_users = list(user_info_df['user_id'].values)
        log(f"总候选用户数量: {len(all_users)}")
        
        # 为了全面评估，从测试集中随机采样更多样本
        sample_size = min(500, len(test_df))  # 增加到500个样本
        test_sample = test_df.sample(n=sample_size, random_state=args.random_state)
        log(f"从测试集中随机抽取 {sample_size} 条记录进行验证")
        
        # 预测结果列表
        predictions = []
        
        # 对测试集的每个样本进行预测
        print("开始生成预测...")
        with alive_bar(len(test_sample), title='生成预测') as bar:
            for idx, row in test_sample.iterrows():
                inviter_id = row['inviter_id']
                item_id = row['item_id']
                timestamp = row['timestamp']
                
                # 推荐top-5 voter
                try:
                    recommended_voters = hybrid_model.predict(inviter_id, item_id, all_users, timestamp)
                    
                    # 添加到预测结果
                    predictions.append({
                        'triple_id': str(idx),
                        'candidate_voter_list': [str(voter) for voter in recommended_voters[:5]]
                    })
                    
                    # 更新进度条
                    bar()
                except Exception as pred_error:
                    # 如果预测失败，随机推荐
                    log(f"预测ID {idx} 失败: {str(pred_error)}, 使用随机推荐")
                    import random
                    random_voters = random.sample(all_users, min(5, len(all_users)))
                    predictions.append({
                        'triple_id': str(idx),
                        'candidate_voter_list': [str(voter) for voter in random_voters]
                    })
                    
                    # 更新进度条
                    bar()
            
        # 保存预测结果
        import json
        with open(os.path.join(args.output_dir, 'results', 'predictions.json'), 'w') as f:
            json.dump(predictions, f)
        
        log(f"预测完成，生成了 {len(predictions)} 个预测结果")
        
        # 如果测试样本中包含真实的voter_id，计算MRR
        if 'voter_id' in test_sample.columns:
            log("计算MRR (Mean Reciprocal Rank)...")
            reciprocal_ranks = []
            
            for idx, pred in enumerate(predictions):
                true_voter = str(test_sample.iloc[idx]['voter_id'])
                candidate_list = pred['candidate_voter_list']
                
                # 计算排名的倒数
                if true_voter in candidate_list:
                    rank = candidate_list.index(true_voter) + 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0)
            
            # 计算MRR
            mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
            log(f"测试样本上的MRR: {mrr:.4f}")
            
            # 保存评估结果
            with open(os.path.join(args.output_dir, 'results', 'evaluation.txt'), 'w') as f:
                f.write(f"MRR: {mrr}\n")
                f.write(f"Sample size: {sample_size}\n")
        
    except Exception as e:
        log(f"预测失败: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return
    
    # 记录总运行时间
    end_time = time.time()
    run_time = end_time - start_time
    log(f"实验完成，总运行时间: {run_time:.2f}秒 ({run_time/60:.2f}分钟)")
    print(f"\n实验完成 ✓")
    print(f"总运行时间: {run_time:.2f}秒 ({run_time/60:.2f}分钟)")
    print(f"结果和日志已保存到: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='动态社交图谱链接预测 - GNN+LSTM+XGBoost模型')
    
    # 数据参数
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--use_sampling', action='store_true',
                        help='是否对训练数据进行采样以减少内存使用')
    parser.add_argument('--sampling_rate', type=float, default=0.3,
                        help='数据采样比例，默认使用30%的数据')
    
    # 模型参数 
    parser.add_argument('--gnn_type', type=str, choices=['gcn', 'sage', 'gat'], default='sage',
                        help='GNN类型：GCN, GraphSAGE或GAT')
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='GNN隐藏层维度')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='节点嵌入维度')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout比率')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--use_xgboost', action='store_true',
                        help='是否使用XGBoost')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机数种子')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录')
    
    args = parser.parse_args()
    
    main(args) 