import os
import argparse
import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime
from alive_progress import alive_bar

# 导入自定义模块
from data_preprocessing import preprocess_data, split_data
from feature_engineering import build_features
from model_building import ModelBuilder

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

# 自定义特征选择函数，带进度条
def feature_selection_with_progress(X, y, percentile=30):
    """使用方差和F值进行特征选择，并显示进度"""
    from sklearn.feature_selection import SelectPercentile, f_classif
    
    print(f"计算每个特征的F值...")
    # 特征数量
    n_features = X.shape[1]
    
    # 如果X是稀疏矩阵，转换为数组会使用大量内存，所以需要分批处理
    batch_size = 100  # 每批处理的特征数
    total_batches = (n_features + batch_size - 1) // batch_size
    
    # 创建数组存储F值和p值
    f_values = np.zeros(n_features)
    p_values = np.zeros(n_features)
    
    # 批量计算F值
    print(f"总共需要处理 {total_batches} 批特征")
    for i in range(0, n_features, batch_size):
        batch_end = min(i + batch_size, n_features)
        batch_features = slice(i, batch_end)
        
        # 进度显示
        simple_progress_bar(i // batch_size + 1, total_batches,
                      prefix=f'特征选择进度:', length=40)
        
        # 只需要处理当前批次的特征
        X_batch = X[:, batch_features]
        
        # 如果X_batch是稀疏矩阵，转换为密集矩阵，但只针对当前批次
        if scipy_sparse and scipy_sparse.issparse(X_batch):
            X_batch = X_batch.toarray()
        
        # 计算当前批次的F值
        f_batch, p_batch = f_classif(X_batch, y)
        
        # 存储结果
        f_values[batch_features] = f_batch
        p_values[batch_features] = p_batch
    
    # 最终进度100%
    simple_progress_bar(total_batches, total_batches,
                  prefix=f'特征选择进度:', length=40)
    
    # 根据F值降序排序获取索引
    indices = np.argsort(f_values)[::-1]
    
    # 选择前percentile%的特征
    threshold = max(1, int(n_features * percentile / 100))
    selected_indices = indices[:threshold]
    
    # 打印一些信息
    print(f"\n选定了 {len(selected_indices)} 个特征 ({percentile}%)")
    
    # 如果X是稀疏矩阵，使用索引选择特征
    if scipy_sparse and scipy_sparse.issparse(X):
        X_selected = X[:, selected_indices]
    else:
        X_selected = X[:, selected_indices]
    
    return X_selected, selected_indices

def main(args):
    start_time = time.time()
    
    # 导入sparse模块 - 全局导入以便在feature_selection_with_progress中使用
    global scipy_sparse
    try:
        import scipy.sparse as scipy_sparse
    except ImportError:
        scipy_sparse = None
    
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
        
        # 保存预处理后的数据
        processed_data_dir = os.path.join(args.output_dir, 'processed_data')
        print("保存预处理后的数据...")
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
        print("正在划分训练集和测试集...")
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
        print("正在为训练集构建特征...")
        train_features = build_features(train_df, user_info_df, item_info_df)
        print("正在为测试集构建特征...")
        test_features = build_features(test_df, user_info_df, item_info_df)
        
        # 保存特征数据
        print("保存特征数据...")
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
        print("准备训练数据...")
        X_train, X_val, y_train, y_val, feature_names = model_builder.prepare_data(train_features, test_size=args.val_size)
        
        log(f"准备训练数据完成，特征数量: {len(feature_names)}")
        print(f"特征矩阵形状: X_train: {X_train.shape}, X_val: {X_val.shape}")
        
        # 内存估算
        try:
            print(f"特征矩阵内存使用: X_train: {X_train.data.nbytes / 1024 / 1024:.2f} MB")
        except:
            print("无法估算特征矩阵内存使用")
            
        # 内存优化: 主动清理不需要的数据
        import gc
        del train_features
        gc.collect()
        
        # 特征选择 - 使用自定义函数带进度条
        print("执行特征选择以减少内存占用...")
        feature_selection_start = time.time()
        # 使用自定义函数进行特征选择
        X_train_selected, selected_indices = feature_selection_with_progress(X_train, y_train, percentile=args.feature_percentile)
        X_val_selected = X_val[:, selected_indices]
        feature_selection_end = time.time()
        print(f"特征选择完成，耗时: {feature_selection_end - feature_selection_start:.2f}秒")
        print(f"选择特征后的形状: X_train_selected: {X_train_selected.shape}, X_val_selected: {X_val_selected.shape}")
        
        # 构建各类模型
        models_to_train = args.models
        print("\n开始训练模型...")
        
        # 单独处理每个模型，避免嵌套进度条
        for i, model_name in enumerate(models_to_train):
            print(f"\n[{i+1}/{len(models_to_train)}] {model_name.upper()}模型...")
            
            # 在每个模型训练前清理内存
            gc.collect()
            
            if model_name == 'rf':
                log("训练随机森林模型...")
                print("正在训练随机森林模型...", end="", flush=True)
                print(f"\n注意: 选用特征数量减少到 {X_train_selected.shape[1]}，可以减轻内存压力")
                try:
                    model_start_time = time.time()
                    # 直接使用已经选择过的特征，不传递额外参数
                    model_builder.build_random_forest(X_train_selected, y_train)
                    model_end_time = time.time()
                    print(f" 完成! (耗时: {model_end_time - model_start_time:.2f}秒)")
                    
                    print("正在评估随机森林模型...", end="", flush=True)
                    eval_start_time = time.time()
                    model_builder.evaluate_model('RandomForest', X_val_selected, y_val)
                    eval_end_time = time.time()
                    print(f" 完成! (耗时: {eval_end_time - eval_start_time:.2f}秒)")
                except MemoryError as me:
                    log(f"随机森林模型训练内存不足: {str(me)}")
                    print(f"\n内存不足，无法训练随机森林模型。请尝试减少特征数量或估计器数量，或增加系统内存。")
                except Exception as e:
                    log(f"随机森林模型训练失败: {str(e)}")
                    print(f"\n随机森林模型训练失败: {str(e)}")
            
            elif model_name == 'gbdt':
                log("训练GBDT模型...")
                print("正在训练GBDT模型...")
                print(f"\n注意: 使用选定的特征子集来避免内存不足问题")
                try:
                    model_start_time = time.time()
                    # 使用选定的特征子集
                    model_builder.build_gbdt(X_train_selected, y_train)
                    model_end_time = time.time()
                    print(f" 完成! (耗时: {model_end_time - model_start_time:.2f}秒)")
                    
                    print("正在评估GBDT模型...", end="", flush=True)
                    eval_start_time = time.time()
                    model_builder.evaluate_model('GBDT', X_val_selected, y_val)
                    eval_end_time = time.time()
                    print(f" 完成! (耗时: {eval_end_time - eval_start_time:.2f}秒)")
                except Exception as e:
                    log(f"GBDT模型训练失败: {str(e)}")
                    print(f"\nGBDT模型训练失败: {str(e)}")
            
            elif model_name == 'lgb':
                log("训练LightGBM模型...")
                print("正在训练LightGBM模型...", end="", flush=True)
                try:
                    model_start_time = time.time()
                    # LightGBM原生支持稀疏矩阵，可以直接使用全部特征
                    model_builder.build_lightgbm(X_train, y_train)
                    model_end_time = time.time()
                    print(f" 完成! (耗时: {model_end_time - model_start_time:.2f}秒)")
                    
                    print("正在评估LightGBM模型...", end="", flush=True)
                    eval_start_time = time.time()
                    model_builder.evaluate_model('LightGBM', X_val, y_val)
                    eval_end_time = time.time()
                    print(f" 完成! (耗时: {eval_end_time - eval_start_time:.2f}秒)")
                except Exception as e:
                    log(f"LightGBM模型训练失败: {str(e)}")
                    print(f"\nLightGBM模型训练失败: {str(e)}")
            
            elif model_name == 'xgb':
                log("训练XGBoost模型...")
                print("正在训练XGBoost模型...", end="", flush=True)
                try:
                    model_start_time = time.time()
                    # XGBoost也支持稀疏矩阵，可以直接使用全部特征
                    model_builder.build_xgboost(X_train, y_train)
                    model_end_time = time.time()
                    print(f" 完成! (耗时: {model_end_time - model_start_time:.2f}秒)")
                    
                    print("正在评估XGBoost模型...", end="", flush=True)
                    eval_start_time = time.time()
                    model_builder.evaluate_model('XGBoost', X_val, y_val)
                    eval_end_time = time.time()
                    print(f" 完成! (耗时: {eval_end_time - eval_start_time:.2f}秒)")
                except Exception as e:
                    log(f"XGBoost模型训练失败: {str(e)}")
                    print(f"\nXGBoost模型训练失败: {str(e)}")
            
            elif model_name == 'dnn':
                log("训练深度神经网络模型...")
                print("正在训练深度神经网络模型...", end="", flush=True)
                try:
                    model_start_time = time.time()
                    # 使用选定的特征子集
                    model_builder.build_dnn(X_train_selected, y_train)
                    model_end_time = time.time()
                    print(f" 完成! (耗时: {model_end_time - model_start_time:.2f}秒)")
                    
                    print("正在评估深度神经网络模型...", end="", flush=True)
                    eval_start_time = time.time()
                    model_builder.evaluate_model('DNN', X_val_selected, y_val)
                    eval_end_time = time.time()
                    print(f" 完成! (耗时: {eval_end_time - eval_start_time:.2f}秒)")
                except Exception as e:
                    log(f"DNN模型训练失败: {str(e)}")
                    print(f"\nDNN模型训练失败: {str(e)}")
            
            elif model_name == 'lstm':
                log("训练LSTM模型...")
                print("正在训练LSTM模型...", end="", flush=True)
                try:
                    model_start_time = time.time()
                    # 使用选定的特征子集
                    model_builder.build_lstm(X_train_selected, y_train, X_val_selected, y_val)
                    model_end_time = time.time()
                    print(f" 完成! (耗时: {model_end_time - model_start_time:.2f}秒)")
                    
                    print("正在评估LSTM模型...", end="", flush=True)
                    eval_start_time = time.time()
                    model_builder.evaluate_model('LSTM', X_val_selected, y_val)
                    eval_end_time = time.time()
                    print(f" 完成! (耗时: {eval_end_time - eval_start_time:.2f}秒)")
                except Exception as e:
                    log(f"LSTM模型训练失败: {str(e)}")
                    print(f"\nLSTM模型训练失败: {str(e)}")
            
            # 显示进度
            print(f"模型 {i+1}/{len(models_to_train)} 完成")
            print("-" * 50)
        
        # 保存模型
        models_dir = os.path.join(args.output_dir, 'models')
        print("\n保存所有模型...", end="", flush=True)
        save_start_time = time.time()
        model_builder.save_models(output_dir=models_dir)
        save_end_time = time.time()
        print(f" 完成! (耗时: {save_end_time - save_start_time:.2f}秒)")
        print(f"所有模型已保存到目录: {models_dir}")
        log(f"模型保存到目录: {models_dir}")
        
        # 在测试集上进行预测
        log("在测试集上进行预测...")
        print("\n在测试集上进行预测...", end="", flush=True)
        predict_start_time = time.time()
        results = model_builder.predict_voter_ids(test_features, top_k=args.top_k)
        predict_end_time = time.time()
        print(f" 完成! (耗时: {predict_end_time - predict_start_time:.2f}秒)")
        print(f"预测结果数量: {len(results)}")
        
        # 保存预测结果
        results_path = os.path.join(args.output_dir, 'results', 'predictions.csv')
        print(f"保存预测结果...", end="", flush=True)
        results.to_csv(results_path, index=False)
        print(" 完成!")
        
        # 计算模型的top-k准确率
        log("计算top-k准确率...")
        print("\n计算Top-K准确率...")
        true_voter_ids = test_features['voter_id'].values
        
        # 使用单个进度条
        for k in [1, 3, 5, 10]:
            if k > args.top_k:
                continue
                
            # 对每个测试样本，获取top-k预测结果
            print(f"\n计算Top-{k}准确率...", end="", flush=True)
            accuracy_start_time = time.time()
            
            top_k_results = results[results['rank'] <= k]
            correct_predictions = 0
            total_samples = len(true_voter_ids)
            
            # 使用一个进度条来显示处理进度
            with alive_bar(total_samples, title=f'Top-{k}评估') as bar:
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
                    
                    # 更新进度条
                    bar()
            
            accuracy = correct_predictions / total_samples
            accuracy_end_time = time.time()
            print(f" 完成! (耗时: {accuracy_end_time - accuracy_start_time:.2f}秒)")
            print(f"Top-{k} 准确率: {accuracy:.4f} ({correct_predictions}/{total_samples})")
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
    print(f"\n实验完成 ✓")
    print(f"总运行时间: {run_time:.2f}秒 ({run_time/60:.2f}分钟)")
    print(f"结果和日志已保存到: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='动态社交图谱链接预测')
    
    # 数据参数
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='验证集比例')
    
    # 模型参数 
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['rf', 'gbdt', 'lgb', 'xgb', 'dnn', 'lstm'],
                        help='要训练的模型列表')
    parser.add_argument('--top_k', type=int, default=10,
                        help='预测top-k个候选结果')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机数种子')
    
    # 随机森林特定参数
    parser.add_argument('--n_estimators', type=int, default=50,
                        help='随机森林中决策树的数量')
    parser.add_argument('--max_features', type=float, default=0.3,
                        help='每棵树考虑的特征比例，可以是float(比例)或int(特征数量)')
    parser.add_argument('--max_depth', type=int, default=20,
                        help='决策树的最大深度')
    parser.add_argument('--use_feature_selection', action='store_true',
                        help='是否使用特征选择减少特征数量')
    parser.add_argument('--feature_percentile', type=int, default=30,
                        help='特征选择时保留的特征百分比')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录')
    
    args = parser.parse_args()
    
    main(args) 