# 社交图谱动态链接预测系统

本项目针对电商平台中的社交图谱动态链接预测问题，基于用户商品分享行为数据，预测在已知邀请用户、商品和时间的情况下，对应的回流用户。

## 项目结构

```
├── data_preprocessing.py    # 数据预处理模块
├── feature_engineering.py   # 特征工程模块
├── model_building.py        # 模型构建模块
├── predict.py               # 预测模块
├── main.py                  # 主程序
├── requirements.txt         # 项目依赖
├── train/                   # 训练数据目录
│   └── preliminary/         # 初赛数据
│       ├── item_share_train_info.json  # 用户动态商品分享数据
│       ├── user_info.json              # 用户信息数据
│       └── item_info.json              # 商品信息数据
├── test/                    # 测试数据目录
└── output/                  # 输出目录
    ├── models/              # 模型保存目录
    ├── results/             # 结果保存目录
    └── processed_data/      # 处理后的数据保存目录
```

## 数据说明

- **item_share_train_info.json**: 用户动态商品分享数据，包含邀请用户id、分享商品id、被邀请用户id、时间戳等信息
- **user_info.json**: 用户信息数据，包含用户id、性别、年龄段、用户等级等信息
- **item_info.json**: 商品信息数据，包含商品id、商品类目id等信息

## 特征工程

本项目构建了多种特征，包括：

1. **用户特征**：
   - 用户基本属性（性别、年龄段、等级）
   - RFM指标（最近活跃时间、活跃频率）
   - 社交活跃度（邀请次数、被邀请次数）
   - 用户商品偏好

2. **商品特征**：
   - 商品基本属性（类目）
   - 商品流行度（被分享次数）
   - 商品转化率（查看者/分享者比率）

3. **用户-商品交互特征**：
   - 交互频率
   - 用户对商品的兴趣度

4. **时间特征**：
   - 时间段划分（上午、下午、晚上、凌晨）
   - 周末/工作日区分
   - 季节性特征

5. **图结构特征**：
   - 节点度（用户连接数）
   - 中心性指标（如PageRank值）
   - 社区结构

## 模型

项目实现了多种机器学习和深度学习模型：

- 逻辑回归（Logistic Regression）
- 随机森林（Random Forest）
- 梯度提升决策树（GBDT）
- LightGBM
- XGBoost
- 深度神经网络（DNN）
- 长短期记忆网络（LSTM）

最终使用模型融合方法，综合多个模型的预测结果，提高预测准确率。

## 安装与运行

### 环境要求

- Python 3.8+
- 依赖包见 requirements.txt

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行方法

**完整的训练和评估流程:**

```bash
python main.py --models lr rf gbdt lgb xgb dnn lstm --top_k 10 --output_dir output
```

**可选参数:**

- `--test_size`: 测试集比例，默认0.2
- `--val_size`: 验证集比例，默认0.2
- `--models`: 要训练的模型列表，可选 lr, rf, gbdt, lgb, xgb, dnn, lstm
- `--top_k`: 预测top-k个候选结果，默认10
- `--random_state`: 随机数种子，默认42
- `--output_dir`: 输出目录，默认output

**只进行预测:**

```bash
python predict.py --test_file test/preliminary/test_data.json --output_file results/predictions.json --models_dir models --top_k 5
```

## 评估指标

模型的性能主要通过以下指标评估：

- **Top-k准确率**: 真实voter_id出现在模型预测的前k个结果中的比例
- **F1 Score**: 精确率和召回率的调和平均值
- **ROC AUC**: 评估模型的区分能力

## 参考文献

1. 吕泽宇,李纪旋,陈如剑,等.电商平台用户再购物行为的预测研究[J].计算机科学,2020,47(S1):424-428.
2. 张诗晨.基于机器学习的电商在线消费者购买行为预测研究[D].吉林大学,2019.
3. Zhang J, Zheng H, Liu J, et al. Research on factors influencing the consumer repurchase intention: Data mining of consumers' online reviews based on machine learning[J]. Neural Computing and Applications, 2024, 36(17):9837-9848. 