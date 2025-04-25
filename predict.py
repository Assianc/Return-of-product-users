import pandas as pd
import numpy as np
import os
import pickle
import argparse
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import xgboost as xgb
import lightgbm as lgb

from data_preprocessing import preprocess_data
from feature_engineering import build_features

def load_models(models_dir='models'):
    """
    加载已经训练好的模型
    """
    models = {}
    
    # 加载sklearn模型
    for model_name in ['LogisticRegression', 'RandomForest', 'GBDT']:
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
    
    # 加载XGBoost模型
    xgb_model_path = os.path.join(models_dir, "XGBoost.pkl")
    if os.path.exists(xgb_model_path):
        with open(xgb_model_path, 'rb') as f:
            models['XGBoost'] = pickle.load(f)
    
    # 加载LightGBM模型
    lgb_model_path = os.path.join(models_dir, "LightGBM.pkl")
    if os.path.exists(lgb_model_path):
        with open(lgb_model_path, 'rb') as f:
            models['LightGBM'] = pickle.load(f)
    
    # 加载Keras模型
    for model_name in ['DNN', 'LSTM']:
        model_path = os.path.join(models_dir, f"{model_name}.h5")
        if os.path.exists(model_path):
            models[model_name] = load_model(model_path)
    
    # 加载标签编码器和特征缩放器
    with open(os.path.join(models_dir, "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(os.path.join(models_dir, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
    
    return models, label_encoder, scaler

def prepare_test_features(test_df, user_info_df, item_info_df, scaler, exclude_cols=None):
    """
    准备测试数据的特征
    """
    if exclude_cols is None:
        exclude_cols = ['timestamp', 'inviter_id', 'item_id', 'voter_id', 
                        'user_id', 'user_id_voter', 'user_id_inviter_graph', 'user_id_voter_graph',
                        'time_period']
    
    # 构建特征
    test_features = build_features(test_df, user_info_df, item_info_df)
    
    # 选择要使用的特征列
    feature_cols = [col for col in test_features.columns if col not in exclude_cols]
    X = test_features[feature_cols].copy()
    
    # 处理缺失值
    X.fillna(0, inplace=True)
    
    # 处理分类特征
    cat_cols = ['user_gender', 'user_age', 'user_level', 
                'user_gender_voter', 'user_age_voter', 'user_level_voter',
                'cate_id']
    cat_cols = [col for col in cat_cols if col in X.columns]
    
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # 标准化数值特征
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    return X, test_features

def ensemble_predict(models, X, label_encoder, weights=None):
    """
    集成多个模型的预测结果
    """
    predictions = {}
    for model_name, model in models.items():
        if model_name in ['LogisticRegression', 'RandomForest', 'GBDT']:
            predictions[model_name] = model.predict_proba(X)
        elif model_name == 'LightGBM':
            predictions[model_name] = model.predict(X)
        elif model_name == 'XGBoost':
            ddata = xgb.DMatrix(X)
            predictions[model_name] = model.predict(ddata)
        elif model_name == 'DNN':
            predictions[model_name] = model.predict(X)
        elif model_name == 'LSTM':
            X_3d = X.values.reshape(X.shape[0], 1, X.shape[1])
            predictions[model_name] = model.predict(X_3d)
    
    # 如果没有提供权重，则使用均等权重
    if weights is None:
        weights = {model_name: 1/len(models) for model_name in models}
    
    # 计算加权平均预测概率
    ensemble_proba = np.zeros((X.shape[0], len(label_encoder.classes_)))
    for model_name, proba in predictions.items():
        ensemble_proba += weights[model_name] * proba
    
    # 返回预测的类别和对应的概率
    pred_class_indices = np.argmax(ensemble_proba, axis=1)
    pred_classes = label_encoder.inverse_transform(pred_class_indices)
    
    return pred_classes, ensemble_proba

def predict_top_k(models, test_features, scaler, label_encoder, top_k=5, exclude_cols=None):
    """
    预测top-k个可能的voter_id
    """
    if exclude_cols is None:
        exclude_cols = ['timestamp', 'inviter_id', 'item_id', 'voter_id', 
                        'user_id', 'user_id_voter', 'user_id_inviter_graph', 'user_id_voter_graph',
                        'time_period']
    
    # 准备特征
    feature_cols = [col for col in test_features.columns if col not in exclude_cols]
    X = test_features[feature_cols].copy()
    
    # 处理缺失值
    X.fillna(0, inplace=True)
    
    # 处理分类特征
    cat_cols = ['user_gender', 'user_age', 'user_level', 
                'user_gender_voter', 'user_age_voter', 'user_level_voter',
                'cate_id']
    cat_cols = [col for col in cat_cols if col in X.columns]
    
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # 标准化数值特征
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    # 使用集成模型预测
    _, ensemble_proba = ensemble_predict(models, X, label_encoder)
    
    # 获取每个样本的top_k个预测结果
    top_k_indices = np.argsort(-ensemble_proba, axis=1)[:, :top_k]
    top_k_probs = np.take_along_axis(ensemble_proba, top_k_indices, axis=1)
    
    # 转换为原始的voter_id
    top_k_voter_ids = label_encoder.inverse_transform(top_k_indices.flatten()).reshape(top_k_indices.shape)
    
    # 构建结果DataFrame
    results = []
    for i in range(len(test_features)):
        for k in range(top_k):
            results.append({
                'inviter_id': test_features.iloc[i]['inviter_id'],
                'item_id': test_features.iloc[i]['item_id'],
                'timestamp': test_features.iloc[i]['timestamp'],
                'pred_voter_id': top_k_voter_ids[i, k],
                'probability': top_k_probs[i, k],
                'rank': k + 1
            })
    
    results_df = pd.DataFrame(results)
    return results_df

def main():
    parser = argparse.ArgumentParser(description='预测voter_id')
    parser.add_argument('--test_file', type=str, default='test/preliminary/test_data.json',
                        help='测试数据文件路径')
    parser.add_argument('--output_file', type=str, default='results/predictions.json',
                        help='输出预测结果文件路径')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='模型保存目录')
    parser.add_argument('--top_k', type=int, default=5,
                        help='返回前k个预测结果')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    
    # 加载预处理后的数据
    # 如果测试数据不包含真实的voter_id，需要特殊处理
    try:
        # 尝试获取原始训练数据
        item_share_df, user_info_df, item_info_df = preprocess_data()
    except:
        print("原始训练数据加载失败，尝试从保存的文件中加载")
        user_info_df = pd.read_csv('processed_data/user_info.csv')
        item_info_df = pd.read_csv('processed_data/item_info.csv')
        item_share_df = None
    
    # 加载模型
    print("加载模型...")
    models, label_encoder, scaler = load_models(args.models_dir)
    
    # 加载测试数据
    print(f"加载测试数据: {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_df = pd.DataFrame(test_data)
    
    # 如果测试数据没有voter_id字段，为了处理方便，添加一个虚拟的字段
    if 'voter_id' not in test_df.columns:
        test_df['voter_id'] = 'unknown'  # 添加一个占位符
    
    # 构建测试数据的特征
    print("构建特征...")
    test_features = build_features(test_df, user_info_df, item_info_df)
    
    # 预测top-k结果
    print(f"预测top-{args.top_k}结果...")
    results = predict_top_k(models, test_features, scaler, label_encoder, args.top_k)
    
    # 保存结果
    print(f"保存预测结果到: {args.output_file}")
    
    # 将结果转换为需要的格式
    # 例如: [{"inviter_id": "xxx", "item_id": "xxx", "timestamp": "xxx", "voter_id": "xxx"}, ...]
    predictions = []
    for inviter_id, group in results[results['rank'] == 1].groupby('inviter_id'):
        for _, row in group.iterrows():
            predictions.append({
                'inviter_id': row['inviter_id'],
                'item_id': row['item_id'],
                'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'voter_id': row['pred_voter_id']
            })
    
    # 保存为JSON格式
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False)
    
    # 如果测试数据有真实的voter_id，计算准确率
    if 'voter_id' in test_df.columns and test_df['voter_id'].iloc[0] != 'unknown':
        print("计算准确率...")
        true_voter_ids = test_df['voter_id'].values
        
        for k in [1, 3, 5, 10]:
            if k > args.top_k:
                continue
            
            # 对每个测试样本，获取top-k预测结果
            top_k_results = results[results['rank'] <= k]
            correct_predictions = 0
            
            # 计算有多少样本的真实voter_id在top-k预测中
            for idx, row in enumerate(true_voter_ids):
                inviter_id = test_df.iloc[idx]['inviter_id']
                item_id = test_df.iloc[idx]['item_id']
                timestamp = test_df.iloc[idx]['timestamp']
                
                # 找到当前样本的预测结果
                sample_preds = top_k_results[
                    (top_k_results['inviter_id'] == inviter_id) & 
                    (top_k_results['item_id'] == item_id) &
                    (top_k_results['timestamp'] == timestamp)
                ]
                
                # 检查真实voter_id是否在预测中
                if row in sample_preds['pred_voter_id'].values:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(true_voter_ids)
            print(f"Top-{k} 准确率: {accuracy:.4f}")
    
    print("预测完成！")

if __name__ == "__main__":
    main() 