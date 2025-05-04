import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy import sparse

# 设置随机种子，确保结果可复现
np.random.seed(42)

class ModelBuilder:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.model_performances = {}
        self.feature_importances = {}
        self.scaler = StandardScaler()
        
    def prepare_data(self, features_df, test_size=0.2):
        """
        准备训练数据，避免重复创建特征
        """
        # 分离特征和标签
        X = features_df.drop(['voter_id'], axis=1)
        y = features_df['voter_id']
        
        # 处理分类特征
        cat_cols = ['user_gender', 'user_age', 'user_level', 
                    'user_gender_voter', 'user_age_voter', 'user_level_voter',
                    'cate_id']
        cat_cols = [col for col in cat_cols if col in X.columns]
        
        # 记录所有特征名称
        feature_names = []
        
        if cat_cols:
            # 使用稀疏矩阵进行独热编码
            sparse_features = []
            for col in cat_cols:
                # 获取唯一值
                unique_values = X[col].unique()
                # 创建映射字典
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                
                # 使用稀疏矩阵直接创建
                row_indices = []
                col_indices = []
                for i, val in enumerate(X[col]):
                    if val in value_map:
                        row_indices.append(i)
                        col_indices.append(value_map[val])
                
                # 创建稀疏矩阵
                data = np.ones(len(row_indices), dtype=np.float32)
                sparse_matrix = sparse.csr_matrix(
                    (data, (row_indices, col_indices)),
                    shape=(len(X), len(unique_values))
                )
                sparse_features.append(sparse_matrix)
                
                # 添加特征名称
                feature_names.extend([f"{col}_{val}" for val in unique_values])
            
            # 合并稀疏特征
            X_sparse = sparse.hstack(sparse_features)
            
            # 处理数值特征
            num_cols = [col for col in X.columns if col not in cat_cols]
            if num_cols:
                X_num = X[num_cols].copy()
                
                # 处理日期时间特征
                for col in X_num.columns:
                    if X_num[col].dtype == 'datetime64[ns]':
                        # 将日期时间转换为时间戳（秒）
                        X_num[col] = X_num[col].astype(np.int64) // 10**9
                    elif X_num[col].dtype == 'object':
                        try:
                            X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
                        except:
                            # 如果无法转换为数值，则将该列视为分类特征
                            cat_cols.append(col)
                            num_cols.remove(col)
                
                # 填充缺失值
                X_num.fillna(X_num.mean(), inplace=True)
                
                # 标准化数值特征
                X_num = pd.DataFrame(self.scaler.fit_transform(X_num), columns=num_cols)
                X_num_sparse = sparse.csr_matrix(X_num)
                
                # 合并所有特征
                X = sparse.hstack([X_sparse, X_num_sparse])
                
                # 添加数值特征名称
                feature_names.extend(num_cols)
            else:
                X = X_sparse
        else:
            # 如果没有分类特征，直接处理数值特征
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
            feature_names = X.columns.tolist()
        
        # 检查并处理稀疏矩阵中的NaN值
        if sparse.issparse(X):
            # 使用稀疏矩阵的方法处理NaN值
            X = X.tocoo()
            # 将NaN值替换为0
            X.data = np.nan_to_num(X.data, nan=0.0)
            # 转回CSR格式
            X = X.tocsr()
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        return X_train, X_val, y_train, y_val, feature_names
    
    def build_logistic_regression(self, X_train, y_train):
        """
        构建逻辑回归模型
        """
        # 使用稀疏矩阵版本的逻辑回归
        model = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            random_state=self.random_state,
            n_jobs=-1,
            solver='liblinear',  # 使用liblinear求解器，支持稀疏矩阵
            multi_class='ovr'    # 使用one-vs-rest策略
        )
        model.fit(X_train, y_train)
        self.models['LogisticRegression'] = model
        return model
    
    def build_random_forest(self, X_train, y_train):
        """
        构建随机森林模型
        """
        # 如果是稀疏矩阵，转换为密集矩阵
        if sparse.issparse(X_train):
            X_train = X_train.toarray()
        
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['RandomForest'] = model
        
        # 记录特征重要性
        self.feature_importances['RandomForest'] = pd.Series(
            model.feature_importances_, 
            index=range(X_train.shape[1])
        ).sort_values(ascending=False)
        
        return model
    
    def build_gbdt(self, X_train, y_train):
        """
        构建梯度提升决策树模型
        """
        # 如果是稀疏矩阵，转换为密集矩阵
        if sparse.issparse(X_train):
            X_train = X_train.toarray()
        
        model = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['GBDT'] = model
        
        # 记录特征重要性
        self.feature_importances['GBDT'] = pd.Series(
            model.feature_importances_, 
            index=range(X_train.shape[1])
        ).sort_values(ascending=False)
        
        return model
    
    def build_lightgbm(self, X_train, y_train):
        """
        构建LightGBM模型
        """
        # 如果是稀疏矩阵，转换为密集矩阵
        if sparse.issparse(X_train):
            X_train = X_train.toarray()
        
        params = {
            'objective': 'multiclass',
            'num_class': len(set(y_train)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_data, num_boost_round=1000)
        self.models['LightGBM'] = model
        
        # 记录特征重要性
        self.feature_importances['LightGBM'] = pd.Series(
            model.feature_importance(), 
            index=range(X_train.shape[1])
        ).sort_values(ascending=False)
        
        return model
    
    def build_xgboost(self, X_train, y_train):
        """
        构建XGBoost模型
        """
        # 如果是稀疏矩阵，转换为密集矩阵
        if sparse.issparse(X_train):
            X_train = X_train.toarray()
        
        params = {
            'objective': 'multi:softprob',
            'num_class': len(set(y_train)),
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'seed': self.random_state
        }
        train_data = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(params, train_data, num_boost_round=1000)
        self.models['XGBoost'] = model
        
        # 记录特征重要性
        self.feature_importances['XGBoost'] = pd.Series(
            model.get_score(importance_type='gain'), 
            index=range(X_train.shape[1])
        ).sort_values(ascending=False)
        
        return model
    
    def build_dnn(self, X_train, y_train):
        """
        构建深度神经网络模型
        """
        # 如果是稀疏矩阵，转换为密集矩阵
        if sparse.issparse(X_train):
            X_train = X_train.toarray()
        
        input_dim = X_train.shape[1]
        output_dim = len(set(y_train))
        
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim, activation='softmax'))
        
        model.compile(loss='sparse_categorical_crossentropy', 
                      optimizer=Adam(learning_rate=0.001), 
                      metrics=['accuracy'])
        
        self.models['DNN'] = model
        
        # 训练模型
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model.fit(
            X_train, y_train, 
            epochs=50, 
            batch_size=32, 
            validation_split=0.2, 
            callbacks=[early_stopping],
            verbose=0
        )
        
        return model
    
    def build_lstm(self, X_train, y_train, X_val, y_val):
        """
        构建LSTM模型
        注意：这需要时序数据，可能需要额外的数据处理
        """
        # 如果是稀疏矩阵，转换为密集矩阵
        if sparse.issparse(X_train):
            X_train = X_train.toarray()
        if sparse.issparse(X_val):
            X_val = X_val.toarray()
        
        # 将特征转为3D格式 [samples, time_steps, features]
        # 这里简化处理，将所有特征视为1个时间步
        X_train_3d = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val_3d = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        input_dim = X_train.shape[1]
        output_dim = len(set(y_train))
        
        model = Sequential()
        model.add(LSTM(128, input_shape=(1, input_dim), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim, activation='softmax'))
        
        model.compile(loss='sparse_categorical_crossentropy', 
                      optimizer=Adam(learning_rate=0.001), 
                      metrics=['accuracy'])
        
        self.models['LSTM'] = model
        
        # 训练模型
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model.fit(
            X_train_3d, y_train, 
            epochs=30, 
            batch_size=32, 
            validation_data=(X_val_3d, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        return model
    
    def evaluate_model(self, model_name, X_val, y_val):
        """
        评估模型性能
        """
        model = self.models[model_name]
        
        # 不同模型预测的方法不同
        if model_name in ['LogisticRegression', 'RandomForest', 'GBDT']:
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)
        elif model_name == 'LightGBM':
            y_pred_proba = model.predict(X_val)
            y_pred = np.argmax(y_pred_proba, axis=1)
        elif model_name == 'XGBoost':
            dval = xgb.DMatrix(X_val)
            y_pred_proba = model.predict(dval)
            y_pred = np.argmax(y_pred_proba, axis=1)
        elif model_name == 'DNN':
            y_pred_proba = model.predict(X_val)
            y_pred = np.argmax(y_pred_proba, axis=1)
        elif model_name == 'LSTM':
            X_val_3d = X_val.values.reshape(X_val.shape[0], 1, X_val.shape[1])
            y_pred_proba = model.predict(X_val_3d)
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 计算各种评估指标
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')
        f1 = f1_score(y_val, y_pred, average='macro')
        
        # 将预测概率重塑为适合ROC AUC计算的格式
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 2:
            # 多分类情况下，使用OVR方法计算ROC AUC
            try:
                roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
            except:
                roc_auc = 0  # 如果计算出错，设为0
        else:
            try:
                roc_auc = roc_auc_score(y_val, y_pred_proba[:, 1])
            except:
                roc_auc = 0
        
        performance = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        self.model_performances[model_name] = performance
        
        print(f"Model {model_name} Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("-" * 50)
        
        return performance
    
    def save_models(self, output_dir='models'):
        """
        保存所有模型
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name in ['DNN', 'LSTM']:
                # Keras模型保存方式不同
                model.save(os.path.join(output_dir, f"{model_name}.h5"))
            else:
                with open(os.path.join(output_dir, f"{model_name}.pkl"), 'wb') as f:
                    pickle.dump(model, f)
        
        # 保存标签编码器和特征缩放器
        with open(os.path.join(output_dir, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        with open(os.path.join(output_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存特征重要性
        for model_name, importance in self.feature_importances.items():
            importance.to_csv(os.path.join(output_dir, f"{model_name}_feature_importance.csv"))
        
        # 保存模型性能指标
        performance_df = pd.DataFrame(self.model_performances).T
        performance_df.to_csv(os.path.join(output_dir, "model_performances.csv"))
        
        print(f"所有模型及相关数据已保存到 {output_dir} 目录")
    
    def ensemble_predict(self, X, weights=None):
        """
        集成多个模型的预测结果
        """
        predictions = {}
        for model_name, model in self.models.items():
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
            weights = {model_name: 1/len(self.models) for model_name in self.models}
        
        # 计算加权平均预测概率
        ensemble_proba = np.zeros((X.shape[0], len(self.label_encoder.classes_)))
        for model_name, proba in predictions.items():
            ensemble_proba += weights[model_name] * proba
        
        # 返回预测的类别和对应的概率
        pred_class_indices = np.argmax(ensemble_proba, axis=1)
        pred_classes = self.label_encoder.inverse_transform(pred_class_indices)
        
        return pred_classes, ensemble_proba
    
    def predict_voter_ids(self, features_df, top_k=5):
        """
        预测voter_ids，并返回每个样本的top_k个候选结果
        """
        # 准备特征
        exclude_cols = ['timestamp', 'inviter_id', 'item_id', 'voter_id', 
                         'user_id', 'user_id_voter', 'user_id_inviter_graph', 'user_id_voter_graph',
                         'time_period']
        
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X = features_df[feature_cols].copy()
        
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
        X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        
        # 使用集成模型预测
        _, ensemble_proba = self.ensemble_predict(X)
        
        # 获取每个样本的top_k个预测结果
        top_k_indices = np.argsort(-ensemble_proba, axis=1)[:, :top_k]
        top_k_probs = np.take_along_axis(ensemble_proba, top_k_indices, axis=1)
        
        # 转换为原始的voter_id
        top_k_voter_ids = self.label_encoder.inverse_transform(top_k_indices.flatten()).reshape(top_k_indices.shape)
        
        # 构建结果DataFrame
        results = []
        for i in range(len(features_df)):
            for k in range(top_k):
                results.append({
                    'inviter_id': features_df.iloc[i]['inviter_id'],
                    'item_id': features_df.iloc[i]['item_id'],
                    'timestamp': features_df.iloc[i]['timestamp'],
                    'pred_voter_id': top_k_voter_ids[i, k],
                    'probability': top_k_probs[i, k],
                    'rank': k + 1
                })
        
        results_df = pd.DataFrame(results)
        return results_df

def main():
    from data_preprocessing import preprocess_data, split_data
    from feature_engineering import build_features
    
    # 创建必要的目录
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 数据预处理
    print("开始数据预处理...")
    item_share_df, user_info_df, item_info_df = preprocess_data()
    
    # 划分训练集和测试集
    print("划分数据集...")
    train_df, val_df = split_data(item_share_df)
    
    # 特征工程
    print("开始特征工程...")
    train_features = build_features(train_df, user_info_df, item_info_df)
    val_features = build_features(val_df, user_info_df, item_info_df)
    
    # 模型构建和训练
    print("开始构建模型...")
    model_builder = ModelBuilder()
    X_train, X_val, y_train, y_val, feature_names = model_builder.prepare_data(train_features)
    
    # 构建各类模型
    print("训练逻辑回归模型...")
    model_builder.build_logistic_regression(X_train, y_train)
    model_builder.evaluate_model('LogisticRegression', X_val, y_val)
    
    print("训练随机森林模型...")
    model_builder.build_random_forest(X_train, y_train)
    model_builder.evaluate_model('RandomForest', X_val, y_val)
    
    print("训练GBDT模型...")
    model_builder.build_gbdt(X_train, y_train)
    model_builder.evaluate_model('GBDT', X_val, y_val)
    
    print("训练LightGBM模型...")
    model_builder.build_lightgbm(X_train, y_train)
    model_builder.evaluate_model('LightGBM', X_val, y_val)
    
    print("训练XGBoost模型...")
    model_builder.build_xgboost(X_train, y_train)
    model_builder.evaluate_model('XGBoost', X_val, y_val)
    
    print("训练深度神经网络模型...")
    model_builder.build_dnn(X_train, y_train)
    model_builder.evaluate_model('DNN', X_val, y_val)
    
    print("训练LSTM模型...")
    model_builder.build_lstm(X_train, y_train, X_val, y_val)
    model_builder.evaluate_model('LSTM', X_val, y_val)
    
    # 保存模型
    model_builder.save_models()
    
    # 在测试集上进行预测
    print("在测试集上进行预测...")
    results = model_builder.predict_voter_ids(val_features)
    results.to_csv('results/predictions.csv', index=False)
    
    # 计算模型的top-k准确率
    print("计算top-k准确率...")
    true_voter_ids = val_features['voter_id'].values
    for k in [1, 3, 5, 10]:
        # 对每个测试样本，获取top-k预测结果
        top_k_results = results[results['rank'] <= k]
        correct_predictions = 0
        
        # 计算有多少样本的真实voter_id在top-k预测中
        for idx, row in enumerate(true_voter_ids):
            inviter_id = val_features.iloc[idx]['inviter_id']
            item_id = val_features.iloc[idx]['item_id']
            timestamp = val_features.iloc[idx]['timestamp']
            
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
    
    print("模型训练与评估完成!")

if __name__ == "__main__":
    main() 