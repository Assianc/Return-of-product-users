"""
预测模块: 包含模型预测和结果生成功能
"""
import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import data_config, gnn_config
from models import GNN, LinkPredictor, HybridModel
from data_utils import load_datasets


def load_trained_models(models_dir=None):
    """
    加载训练好的模型
    
    参数:
        models_dir: 模型目录
    
    返回:
        gnn_model: GNN模型
        link_predictor: 链接预测器
        node_features: 节点特征
        edge_index: 边索引
        node_mapping: 节点ID到索引的映射
    """
    if models_dir is None:
        models_dir = data_config.models_dir
    
    print("加载训练好的模型...")
    
    # 加载节点特征和边索引
    node_features_path = os.path.join(data_config.processed_data_dir, 'node_features.pt')
    edge_index_path = os.path.join(data_config.processed_data_dir, 'edge_index.pt')
    node_mapping_path = os.path.join(data_config.processed_data_dir, 'node_mapping.pkl')
    
    try:
        # 加载数据
        node_features = torch.load(node_features_path)
        edge_index = torch.load(edge_index_path)
        
        with open(node_mapping_path, 'rb') as f:
            node_mapping = pickle.load(f)
            
        print(f"节点特征大小: {node_features.shape}, 边数量: {edge_index.shape[1]//2}")
    except FileNotFoundError:
        raise FileNotFoundError("找不到预处理的图数据文件，请先运行训练")
    except Exception as e:
        raise Exception(f"加载图数据失败: {str(e)}")
    
    # 加载GNN模型
    gnn_model_path = os.path.join(models_dir, 'gnn_model.pt')
    link_predictor_path = os.path.join(models_dir, 'link_predictor.pt')
    
    if not os.path.exists(gnn_model_path) or not os.path.exists(link_predictor_path):
        # 如果没有训练好的模型，返回初始化的模型
        print("找不到保存的模型文件，将使用新初始化的模型")
        
        # 初始化默认模型
        in_channels = node_features.shape[1]
        hidden_channels = gnn_config.hidden_channels
        out_channels = gnn_config.embedding_dim
        
        gnn_model = GNN(in_channels, hidden_channels, out_channels, 
                       dropout=gnn_config.dropout, gnn_type=gnn_config.gnn_type)
        link_predictor = LinkPredictor(out_channels, hidden_channels)
        
        return gnn_model, link_predictor, node_features, edge_index, node_mapping
    
    try:
        # 加载模型状态以检查参数形状
        state_dict = torch.load(gnn_model_path, map_location='cpu')
        
        # 检查conv1.lin_l.weight的形状来确定hidden_channels
        hidden_channels = state_dict['conv1.lin_l.weight'].shape[0]
        
        # 检查conv2.lin_l.weight的形状来确定embedding_dim
        embedding_dim = state_dict['conv2.lin_l.weight'].shape[0]
        
        print(f"从保存的模型中读取参数大小: hidden_channels={hidden_channels}, embedding_dim={embedding_dim}")
        
        # 初始化模型使用检测到的参数大小
        in_channels = node_features.shape[1]
        
        gnn_model = GNN(in_channels, hidden_channels, embedding_dim, 
                     dropout=gnn_config.dropout, gnn_type=gnn_config.gnn_type)
        link_predictor = LinkPredictor(embedding_dim, hidden_channels)
        
        # 加载模型参数
        gnn_model.load_state_dict(state_dict)
        link_predictor.load_state_dict(torch.load(link_predictor_path, map_location='cpu'))
        
        # 设置为评估模式
        gnn_model.eval()
        link_predictor.eval()
        
        print("模型加载完成")
        
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        print("将使用新初始化的模型")
        
        # 初始化默认模型
        in_channels = node_features.shape[1]
        hidden_channels = gnn_config.hidden_channels
        out_channels = gnn_config.embedding_dim
        
        gnn_model = GNN(in_channels, hidden_channels, out_channels, 
                       dropout=gnn_config.dropout, gnn_type=gnn_config.gnn_type)
        link_predictor = LinkPredictor(out_channels, hidden_channels)
    
    return gnn_model, link_predictor, node_features, edge_index, node_mapping


def generate_predictions(test_df, gnn_model, link_predictor, node_features, edge_index, node_mapping, 
                         batch_size=100, top_k=5, use_xgboost=True):
    """
    为测试集生成预测
    
    参数:
        test_df: 测试数据
        gnn_model: GNN模型
        link_predictor: 链接预测器
        node_features: 节点特征
        edge_index: 边索引
        node_mapping: 节点ID到索引的映射
        batch_size: 批处理大小
        top_k: 推荐数量
        use_xgboost: 是否使用XGBoost
        
    返回:
        predictions: 预测结果列表
    """
    print("为测试集生成预测...")
    
    # 创建混合模型
    hybrid_model = HybridModel(gnn_model, link_predictor, node_features, edge_index, node_mapping, use_xgboost=use_xgboost)
    
    # 加载XGBoost模型（如果存在）
    if use_xgboost:
        try:
            hybrid_model.load(data_config.models_dir)
            print("已加载XGBoost模型")
        except Exception as e:
            print(f"加载XGBoost模型失败: {e}, 将仅使用GNN")
            hybrid_model.use_xgboost = False
    
    # 预计算节点嵌入
    hybrid_model.precompute_embeddings()
    
    # 加载用户信息
    _, _, user_info_df, _ = load_datasets()
    
    # 获取所有候选用户
    all_users = list(user_info_df['user_id'].values)
    print(f"总候选用户数量: {len(all_users)}")
    
    # 预测结果列表
    predictions = []
    
    # 对测试集的每个样本进行预测
    print("开始生成预测...")
    with tqdm(total=len(test_df), desc="生成预测") as pbar:
        for idx, row in test_df.iterrows():
            inviter_id = row['inviter_id']
            item_id = row['item_id']
            triple_id = row['triple_id']
            timestamp = row['timestamp'] if 'timestamp' in row else None
            
            try:
                # 推荐top-k voter
                recommended_voters = hybrid_model.predict(
                    inviter_id, 
                    item_id, 
                    all_users, 
                    timestamp=timestamp,
                    batch_size=batch_size
                )
                
                # 确保推荐列表长度为top_k
                if len(recommended_voters) < top_k:
                    # 如果推荐不足，随机添加一些用户
                    import random
                    remaining = [u for u in all_users if u not in recommended_voters]
                    additional = random.sample(remaining, min(top_k - len(recommended_voters), len(remaining)))
                    recommended_voters.extend(additional)
                
                # 添加到预测结果
                predictions.append({
                    'triple_id': str(triple_id),
                    'candidate_voter_list': [str(voter) for voter in recommended_voters[:top_k]]
                })
                
            except Exception as e:
                # 如果预测失败，随机推荐
                print(f"预测ID {triple_id} 失败: {e}, 使用随机推荐")
                import random
                random_voters = random.sample(all_users, min(top_k, len(all_users)))
                predictions.append({
                    'triple_id': str(triple_id),
                    'candidate_voter_list': [str(voter) for voter in random_voters]
                })
            
            # 更新进度条
            pbar.update(1)
    
    print(f"预测完成，生成了 {len(predictions)} 个预测结果")
    
    return predictions


def save_predictions(predictions, output_file=None):
    """
    保存预测结果
    
    参数:
        predictions: 预测结果列表
        output_file: 输出文件路径
    """
    if output_file is None:
        output_file = os.path.join(data_config.results_dir, 'predictions.json')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"保存预测结果到: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(predictions, f)
    
    print("预测结果已保存")


def run_prediction_pipeline(test_file=None, output_file=None, batch_size=100, top_k=5, use_xgboost=True):
    """
    运行完整的预测流程
    
    参数:
        test_file: 测试文件路径
        output_file: 输出文件路径
        batch_size: 批处理大小
        top_k: 推荐数量
        use_xgboost: 是否使用XGBoost
    """
    if test_file is None:
        test_file = data_config.test_file
    
    if output_file is None:
        output_file = os.path.join(data_config.results_dir, 'predictions.json')
    
    print(f"开始预测流程，测试文件: {test_file}")
    
    # 加载测试数据
    test_df = pd.read_json(test_file)
    print(f"测试数据加载完成，共 {len(test_df)} 条记录")
    
    # 加载模型
    gnn_model, link_predictor, node_features, edge_index, node_mapping = load_trained_models()
    
    # 生成预测
    predictions = generate_predictions(
        test_df, 
        gnn_model, 
        link_predictor, 
        node_features, 
        edge_index, 
        node_mapping, 
        batch_size=batch_size, 
        top_k=top_k,
        use_xgboost=use_xgboost
    )
    
    # 保存预测结果
    save_predictions(predictions, output_file)
    
    print("预测流程完成")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行预测流程')
    parser.add_argument('--test_file', type=str, default=None, help='测试文件路径')
    parser.add_argument('--output_file', type=str, default=None, help='输出文件路径')
    parser.add_argument('--batch_size', type=int, default=100, help='批处理大小')
    parser.add_argument('--top_k', type=int, default=5, help='推荐数量')
    parser.add_argument('--use_xgboost', action='store_true', help='是否使用XGBoost')
    
    args = parser.parse_args()
    
    run_prediction_pipeline(
        test_file=args.test_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        top_k=args.top_k,
        use_xgboost=args.use_xgboost
    ) 