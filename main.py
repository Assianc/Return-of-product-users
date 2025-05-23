"""
主入口文件: 整合所有模块，提供命令行接口
"""
import os
import time
import argparse
from datetime import datetime

from config import data_config, gnn_config, hybrid_model_config, training_config
from data_utils import load_datasets, prepare_graph_data_for_training, create_train_val_split, prepare_link_prediction_data
from train import train_all_models, Logger
from predict import run_prediction_pipeline


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='动态社交图谱链接预测 - GNN+LSTM+XGBoost模型')
    
    # 主命令类型
    subparsers = parser.add_subparsers(dest='command', help='指定要执行的操作')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--test_size', type=float, default=data_config.test_size, help='测试集比例')
    train_parser.add_argument('--val_size', type=float, default=data_config.val_size, help='验证集比例')
    train_parser.add_argument('--use_sampling', action='store_true', default=data_config.use_sampling, help='是否对训练数据进行采样')
    train_parser.add_argument('--sampling_rate', type=float, default=data_config.sampling_rate, help='数据采样比例')
    train_parser.add_argument('--gnn_type', type=str, choices=['gcn', 'sage', 'gat'], default=gnn_config.gnn_type, help='GNN类型')
    train_parser.add_argument('--hidden_channels', type=int, default=gnn_config.hidden_channels, help='GNN隐藏层维度')
    train_parser.add_argument('--embedding_dim', type=int, default=gnn_config.embedding_dim, help='节点嵌入维度')
    train_parser.add_argument('--dropout', type=float, default=gnn_config.dropout, help='Dropout比率')
    train_parser.add_argument('--learning_rate', type=float, default=gnn_config.learning_rate, help='学习率')
    train_parser.add_argument('--epochs', type=int, default=gnn_config.epochs, help='训练轮数')
    train_parser.add_argument('--batch_size', type=int, default=gnn_config.batch_size, help='批处理大小')
    train_parser.add_argument('--use_xgboost', action='store_true', default=hybrid_model_config.use_xgboost, help='是否使用XGBoost')
    train_parser.add_argument('--use_cached_data', action='store_true', default=data_config.use_cached_data, help='是否使用缓存数据')
    train_parser.add_argument('--random_state', type=int, default=training_config.random_state, help='随机数种子')
    train_parser.add_argument('--output_dir', type=str, default=data_config.output_dir, help='输出目录')
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='生成预测')
    predict_parser.add_argument('--test_file', type=str, default=data_config.test_file, help='测试文件路径')
    predict_parser.add_argument('--output_file', type=str, default=None, help='输出文件路径')
    predict_parser.add_argument('--batch_size', type=int, default=100, help='批处理大小')
    predict_parser.add_argument('--top_k', type=int, default=5, help='推荐数量')
    predict_parser.add_argument('--use_xgboost', action='store_true', default=hybrid_model_config.use_xgboost, help='是否使用XGBoost')
    
    # 全流程命令
    pipeline_parser = subparsers.add_parser('pipeline', help='执行完整流程(训练+预测)')
    pipeline_parser.add_argument('--test_file', type=str, default=data_config.test_file, help='测试文件路径')
    pipeline_parser.add_argument('--output_file', type=str, default=None, help='输出文件路径')
    pipeline_parser.add_argument('--use_cached_data', action='store_true', default=data_config.use_cached_data, help='是否使用缓存数据')
    pipeline_parser.add_argument('--use_xgboost', action='store_true', default=hybrid_model_config.use_xgboost, help='是否使用XGBoost')
    pipeline_parser.add_argument('--use_sampling', action='store_true', default=data_config.use_sampling, help='是否对训练数据进行采样')
    pipeline_parser.add_argument('--sampling_rate', type=float, default=data_config.sampling_rate, help='数据采样比例')
    
    args = parser.parse_args()
    
    # 根据命令执行不同操作
    if args.command == 'train':
        # 更新配置
        data_config.test_size = args.test_size
        data_config.val_size = args.val_size
        data_config.use_sampling = args.use_sampling
        data_config.sampling_rate = args.sampling_rate
        data_config.output_dir = args.output_dir
        data_config.use_cached_data = args.use_cached_data
        
        gnn_config.gnn_type = args.gnn_type
        gnn_config.hidden_channels = args.hidden_channels
        gnn_config.embedding_dim = args.embedding_dim
        gnn_config.dropout = args.dropout
        gnn_config.learning_rate = args.learning_rate
        gnn_config.epochs = args.epochs
        gnn_config.batch_size = args.batch_size
        
        hybrid_model_config.use_xgboost = args.use_xgboost
        
        training_config.random_state = args.random_state
        
        # 执行训练
        train_all_models(use_cached_data=args.use_cached_data)
        
    elif args.command == 'predict':
        # 执行预测
        run_prediction_pipeline(
            test_file=args.test_file,
            output_file=args.output_file,
            batch_size=args.batch_size,
            top_k=args.top_k,
            use_xgboost=args.use_xgboost
        )
        
    elif args.command == 'pipeline':
        # 执行完整流程
        start_time = time.time()
        
        # 创建必要的目录
        os.makedirs(data_config.output_dir, exist_ok=True)
        os.makedirs(data_config.models_dir, exist_ok=True)
        os.makedirs(data_config.results_dir, exist_ok=True)
        
        # 初始化日志
        log_file = os.path.join(data_config.output_dir, 'experiment_log.txt')
        logger = Logger(log_file)
        
        logger.log("开始执行完整流程...")
        
        # 训练
        logger.log("第1阶段: 训练模型")
        hybrid_model_config.use_xgboost = args.use_xgboost
        data_config.use_cached_data = args.use_cached_data
        data_config.use_sampling = args.use_sampling
        data_config.sampling_rate = args.sampling_rate
        train_all_models(use_cached_data=args.use_cached_data)
        
        # 预测
        logger.log("第2阶段: 生成预测")
        run_prediction_pipeline(
            test_file=args.test_file,
            output_file=args.output_file,
            use_xgboost=args.use_xgboost
        )
        
        # 记录总运行时间
        end_time = time.time()
        run_time = end_time - start_time
        logger.log(f"完整流程执行完毕，总运行时间: {run_time:.2f}秒 ({run_time/60:.2f}分钟)")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 