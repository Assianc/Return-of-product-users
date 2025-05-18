# 动态社交图谱链接预测

## 项目介绍

本项目实现了一个基于GNN和XGBoost的混合模型，用于预测社交网络中的链接关系。该系统能够处理大规模电商社交网络数据，并提供高质量的推荐结果。

主要功能：
- 使用GNN模型捕获网络结构信息  
- 结合XGBoost进行预测优化
- 支持用户-物品交互的二分图建模
- 高效处理大规模数据集
- 动态链接预测和推荐

## 环境要求

```
numpy>=1.22.0
pandas>=1.3.5
scikit-learn>=1.0.1
torch>=1.12.0
torch-geometric>=2.0.0
xgboost>=1.5.1
networkx>=2.6.0
tqdm>=4.62.0
alive-progress>=2.1.0
psutil>=5.9.0
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

运行主程序：

```bash
python main.py --gnn_type sage --use_xgboost --output_dir output
```

### 使用数据采样加速开发和调试

对于大型数据集，可以使用采样功能加快处理速度：

```bash
# 使用5%的数据进行采样
python main.py --use_sampling --sampling_rate 0.05 --output_dir output

# 使用10%的数据并启用XGBoost
python main.py --use_sampling --sampling_rate 0.1 --use_xgboost --output_dir output
```

### 参数说明

- `--test_size`: 测试集比例，默认0.2
- `--val_size`: 验证集比例，默认0.2
- `--use_sampling`: 是否对训练数据进行采样，用于加速开发和调试
- `--sampling_rate`: 数据采样比例，默认0.3，表示使用30%的数据
- `--gnn_type`: GNN类型，可选'gcn', 'sage', 'gat'，默认'sage'
- `--hidden_channels`: GNN隐藏层维度，默认128
- `--embedding_dim`: 节点嵌入维度，默认64
- `--dropout`: Dropout比率，默认0.2
- `--learning_rate`: 学习率，默认0.001
- `--epochs`: 训练轮数，默认100
- `--use_xgboost`: 是否使用XGBoost
- `--random_state`: 随机数种子，默认42
- `--output_dir`: 输出目录，默认'output'

## 输出结果

程序会在指定的输出目录中生成以下内容：
- `models/`: 保存训练好的模型
- `results/`: 保存预测结果和评估指标
- `processed_data/`: 保存预处理后的数据
- `experiment_log.txt`: 实验日志

## 数据格式说明

### 训练数据

训练数据包含三个主要文件：

1. **item_share_train_info.json**: 用户动态商品分享数据
   ```json
   {
     "inviter_id": "用户ID",
     "item_id": "商品ID",
     "voter_id": "被邀请用户ID",
     "timestamp": "时间戳"
   }
   ```

2. **user_info.json**: 用户信息数据
   ```json
   {
     "user_id": "用户ID",
     "user_gender": "用户性别",
     "user_age": "用户年龄段",
     "user_level": "用户等级"
   }
   ```

3. **item_info.json**: 商品信息数据
   ```json
   {
     "item_id": "商品ID",
     "cate_id": "商品类目ID",
     "cate_level1_id": "商品一级类目ID",
     "brand_id": "品牌ID",
     "shop_id": "店铺ID"
   }
   ```

### 测试数据

测试数据格式：**item_share_preliminary_test_info.json**
```json
{
  "triple_id": "测试样本ID",
  "inviter_id": "用户ID",
  "item_id": "商品ID"
}
```

注意：测试数据中没有voter_id字段，需要由模型预测推荐的voter。

### 提交格式

预测结果保存为JSON格式：
```json
[
  {
    "triple_id": "测试样本ID",
    "candidate_voter_list": ["推荐用户ID1", "推荐用户ID2", "推荐用户ID3", "推荐用户ID4", "推荐用户ID5"]
  },
  "..."
]
```

## 适用场景

本项目特别适合以下场景：
- 大规模社交网络推荐系统
- 电商平台的用户行为预测
- 动态网络链接预测
- 社交媒体影响力分析
- 实时推荐系统集成

## 优化和性能

该系统针对大规模数据集进行了多项优化：
- 高效的图数据结构和处理流程
- 并行计算和优化的训练过程
- 内存占用优化
- 完善的进度监控和错误处理
- 灵活的模型参数配置
- 数据采样功能，加速开发和调试过程

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
│   └── preliminary/         # 初赛测试数据
│       └── item_share_preliminary_test_info.json  # 测试数据
└── output/                  # 输出目录
    ├── models/              # 模型保存目录
    ├── results/             # 结果保存目录
    └── processed_data/      # 处理后的数据保存目录
```

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

- 图神经网络(GNN)：包括GCN, GraphSAGE, GAT
- XGBoost
- 混合模型：结合GNN和XGBoost的优势

系统采用了混合模型架构，利用GNN捕获网络拓扑结构信息，同时使用XGBoost处理特征交互和非线性关系，综合提高预测准确率。

## 注意事项

1. **处理大型数据集**：对于大型数据集，建议先使用采样功能进行模型调试，确认无误后再使用完整数据集训练最终模型。

2. **ID类型处理**：系统支持处理不同格式的ID（整数或字符串），会自动将所有ID转换为字符串格式以确保一致性。

3. **内存优化**：如果遇到内存不足的问题，可以尝试以下方法：
   - 降低采样率（如`--sampling_rate 0.01`）
   - 减小模型参数（如`--hidden_channels 64 --embedding_dim 32`）
   - 增加系统交换空间

## 参考文献

1. 吕泽宇,李纪旋,陈如剑,等.电商平台用户再购物行为的预测研究[J].计算机科学,2020,47(S1):424-428.
2. 张诗晨.基于机器学习的电商在线消费者购买行为预测研究[D].吉林大学,2019.
3. Zhang J, Zheng H, Liu J, et al. Research on factors influencing the consumer repurchase intention: Data mining of consumers' online reviews based on machine learning[J]. Neural Computing and Applications, 2024, 36(17):9837-9848.
4. Hamilton W, Ying Z, Leskovec J. Inductive representation learning on large graphs[C]//Advances in neural information processing systems. 2017: 1024-1034.
5. Veličković P, Cucurull G, Casanova A, et al. Graph attention networks[J]. arXiv preprint arXiv:1710.10903, 2017. 