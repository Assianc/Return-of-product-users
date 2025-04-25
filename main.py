import os
import argparse
import pandas as pd
import numpy as np
import time
from datetime import datetime

# 导入自定义模块
from data_preprocessing import preprocess_data, split_data
from feature_engineering import build_features
from model_building import ModelBuilder

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
        item_share_df, user_info_df, item_info_df = preprocess_data()
        
        # 保存预处理后的数据
        processed_data_dir = os.path.join(args.output_dir, 'processed_data')
        user_info_df.to_csv(os.path.join(processed_data_dir, 'user_info.csv'))
        item_info_df.to_csv(os.path.join(processed_data_dir, 'item_info.csv'))
        item_share_df.to_csv(os.path.join(processed_data_dir, 'item_share.csv'), index=False)
        
        log(f"数据预处理完成，商品分享记录数量: {len(item_share_df)}, 用户数量: {len(user_info_df)}, 商品数量: {len(item_info_df)}")
    except Exception as e:
        log(f"数据预处理失败: {str(e)}")
        return
    
    # 划分训练集和测试集
    log("划分数据集...")
    try:
        train_df, test_df = split_data(item_share_df, test_size=args.test_size)
        
        # 保存划分后的数据
        train_df.to_csv(os.path.join(processed_data_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(processed_data_dir, 'test.csv'), index=False)
        
        log(f"数据集划分完成，训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    except Exception as e:
        log(f"数据集划分失败: {str(e)}")
        return
    
    # 特征工程
    log("开始特征工程...")
    try:
        train_features = build_features(train_df, user_info_df, item_info_df)
        test_features = build_features(test_df, user_info_df, item_info_df)
        
        # 保存特征数据
        train_features.to_csv(os.path.join(processed_data_dir, 'train_features.csv'), index=False)
        test_features.to_csv(os.path.join(processed_data_dir, 'test_features.csv'), index=False)
        
        log(f"特征工程完成，训练集特征数量: {train_features.shape[1]}, 测试集特征数量: {test_features.shape[1]}")
    except Exception as e:
        log(f"特征工程失败: {str(e)}")
        return
    
    # 模型构建和训练
    log("开始构建模型...")
    try:
        model_builder = ModelBuilder(random_state=args.random_state)
        X_train, X_val, y_train, y_val, feature_names = model_builder.prepare_data(train_features, test_size=args.val_size)
        
        log(f"准备训练数据完成，特征数量: {len(feature_names)}")
        
        # 构建各类模型
        if 'lr' in args.models:
            log("训练逻辑回归模型...")
            model_builder.build_logistic_regression(X_train, y_train)
            model_builder.evaluate_model('LogisticRegression', X_val, y_val)
        
        if 'rf' in args.models:
            log("训练随机森林模型...")
            model_builder.build_random_forest(X_train, y_train)
            model_builder.evaluate_model('RandomForest', X_val, y_val)
        
        if 'gbdt' in args.models:
            log("训练GBDT模型...")
            model_builder.build_gbdt(X_train, y_train)
            model_builder.evaluate_model('GBDT', X_val, y_val)
        
        if 'lgb' in args.models:
            log("训练LightGBM模型...")
            model_builder.build_lightgbm(X_train, y_train)
            model_builder.evaluate_model('LightGBM', X_val, y_val)
        
        if 'xgb' in args.models:
            log("训练XGBoost模型...")
            model_builder.build_xgboost(X_train, y_train)
            model_builder.evaluate_model('XGBoost', X_val, y_val)
        
        if 'dnn' in args.models:
            log("训练深度神经网络模型...")
            model_builder.build_dnn(X_train, y_train)
            model_builder.evaluate_model('DNN', X_val, y_val)
        
        if 'lstm' in args.models:
            log("训练LSTM模型...")
            model_builder.build_lstm(X_train, y_train, X_val, y_val)
            model_builder.evaluate_model('LSTM', X_val, y_val)
        
        # 保存模型
        models_dir = os.path.join(args.output_dir, 'models')
        model_builder.save_models(output_dir=models_dir)
        log(f"模型保存到目录: {models_dir}")
        
        # 在测试集上进行预测
        log("在测试集上进行预测...")
        results = model_builder.predict_voter_ids(test_features, top_k=args.top_k)
        results.to_csv(os.path.join(args.output_dir, 'results', 'predictions.csv'), index=False)
        
        # 计算模型的top-k准确率
        log("计算top-k准确率...")
        true_voter_ids = test_features['voter_id'].values
        for k in [1, 3, 5, 10]:
            if k > args.top_k:
                continue
                
            # 对每个测试样本，获取top-k预测结果
            top_k_results = results[results['rank'] <= k]
            correct_predictions = 0
            
            # 计算有多少样本的真实voter_id在top-k预测中
            for idx, row in enumerate(true_voter_ids):
                inviter_id = test_features.iloc[idx]['inviter_id']
                item_id = test_features.iloc[idx]['item_id']
                timestamp = test_features.iloc[idx]['timestamp']
                
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
            log(f"Top-{k} 准确率: {accuracy:.4f}")
        
    except Exception as e:
        log(f"模型训练与评估失败: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return
    
    # 记录总运行时间
    end_time = time.time()
    run_time = end_time - start_time
    log(f"实验完成，总运行时间: {run_time:.2f}秒 ({run_time/60:.2f}分钟)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='动态社交图谱链接预测')
    
    # 数据参数
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='验证集比例')
    
    # 模型参数
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['lr', 'rf', 'gbdt', 'lgb', 'xgb', 'dnn', 'lstm'],
                        help='要训练的模型列表')
    parser.add_argument('--top_k', type=int, default=10,
                        help='预测top-k个候选结果')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机数种子')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录')
    
    args = parser.parse_args()
    
    main(args) 