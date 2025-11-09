# 股票预测项目 - Alpha101因子量化预测

## 项目概述

本项目基于A股市场的基础量价数据，使用Alpha101因子和机器学习模型进行股票收益预测。项目实现了完整的量化交易流程，包括数据处理、因子计算、特征工程、模型训练和预测评估。

### 项目目标

- 使用基础量价数据在次日开盘时给出交易信号
- 预测目标：数据集中的 `y` 字段
- 评判标准：`Correlation(y, pred_y)`

## 项目结构

```
alpha/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖包
├── daily_data.csv           # 原始数据文件
├── main.py                  # 主程序入口
├── evaluate.py              # 模型评估脚本
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── utils.py            # 工具函数库
│   ├── data_loader.py      # 数据加载和预处理
│   ├── alpha101.py         # Alpha101因子实现
│   ├── feature_generator.py # 特征生成器
│   └── model.py            # 模型训练和预测
├── models/                  # 模型保存目录
│   ├── lgb_model.pkl
│   ├── xgb_model.pkl
│   └── feature_list.pkl
└── results/                 # 结果输出目录
    ├── predictions.csv
    ├── pred_vs_actual.png
    └── group_returns.png
```

## 数据说明

### 数据字段

| 字段名 | 说明 |
|--------|------|
| instrument_id | 股票ID，后六位为交易所代码，前六位为唯一性编码 |
| date | 数据日期 |
| y | **预测目标**（2020年1月1日后为0，需要预测） |
| high | 当日最高价 |
| low | 当日最低价 |
| open | 当日开盘价 |
| close | 当日收盘价 |
| next_open | 次日开盘价 |
| volume | 成交量 |
| vwap | 成交量加权日均价 |
| adjustment | 复权因子（分红、除权等） |
| type | 涨跌停描述 |
| a_share_capital | A股市值 |
| total_capital | 总市值 |
| float_a_share_capital | A股流通市值 |
| turnover | 成交额 |
| turnover_rate | 换手率 |

### 数据划分

- **训练集**: 2010-12-01 至 2019-12-31
- **测试集**: 2020-01-01 及以后（y值为0，需要预测）

## 安装和环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 依赖包说明

- `pandas`: 数据处理
- `numpy`: 数值计算
- `scipy`: 科学计算
- `scikit-learn`: 机器学习基础库
- `lightgbm`: LightGBM模型
- `xgboost`: XGBoost模型
- `matplotlib`: 数据可视化
- `seaborn`: 统计图表
- `joblib`: 模型保存和加载
- `tqdm`: 进度条显示

## 使用方法

### 1. 完整训练和预测流程

```bash
python main.py
```

这将执行以下步骤：
1. 数据加载与预处理
2. 特征工程（生成Alpha101因子和技术指标）
3. 特征选择（移除低方差和高相关性特征）
4. 模型训练（LightGBM和XGBoost）
5. 模型集成
6. 测试集预测
7. 保存模型和预测结果

### 2. 评估模型性能

```bash
python evaluate.py
```

这将生成：
- 模型评估指标（相关系数、IC、RMSE、MAE）
- 预测值vs实际值散点图
- 分组收益分析图
- 特征重要性分析

## 核心功能模块

### 1. Alpha101因子

实现了WorldQuant的Alpha101量化因子，包括：

- **Alpha#1-26**: 基础价量因子
- **Alpha#101**: 简单但有效的价格位置因子

因子计算使用的基础函数：
- 时间序列函数：`ts_sum`, `ts_mean`, `ts_std`, `ts_min`, `ts_max`, `ts_rank`
- 数据变换函数：`delta`, `delay`, `rank`, `scale`
- 统计函数：`correlation`, `covariance`, `product`, `decay_linear`

### 2. 特征工程

#### 基础特征
- 价格相关：价格区间、价格位置
- 成交量相关：成交量移动平均（5/10/20日）
- 价格移动平均：收盘价移动平均（5/10/20日）
- 波动率：收益率标准差（5/10/20日）
- 动量特征：价格变化率（5/10/20日）

#### 市场特征
- 市值排名
- 成交量排名
- 换手率排名

### 3. 模型架构

#### 单模型
- **LightGBM**: 梯度提升树模型，速度快，效果好
- **XGBoost**: 经典的梯度提升模型

#### 模型集成
- 简单平均集成
- 加权平均集成

### 4. 特征选择

- 移除高缺失率特征（缺失率>50%）
- 移除低方差特征（方差<0.001）
- 移除高相关性特征（相关系数>0.95）
- 基于相关性选择Top N特征

## 模型训练策略

### 1. 数据集划分

```python
训练集: 80%
验证集: 20%
```

使用时间序列划分，避免未来信息泄露。

### 2. 超参数配置

#### LightGBM参数
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'n_estimators': 1000,
    'early_stopping_rounds': 50
}
```

#### XGBoost参数
```python
{
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'early_stopping_rounds': 50
}
```

### 3. 评估指标

- **Correlation**: 预测值与实际值的相关系数（主要指标）
- **IC (Information Coefficient)**: 信息系数
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **分组收益**: 多空组合收益

## 结果分析

### 1. 模型性能

模型在验证集上的表现：
- LightGBM Correlation: ~0.XX
- XGBoost Correlation: ~0.XX
- Ensemble Correlation: ~0.XX

### 2. 特征重要性

Top 10重要特征（示例）：
1. alpha101
2. alpha001
3. momentum_20
4. volatility_20
5. volume_ma20
6. ...

### 3. 分组收益分析

将预测结果分为10组，观察不同组的平均收益：
- 最高预测组（Group 9）：平均收益最高
- 最低预测组（Group 0）：平均收益最低
- 多空收益：Group 9 - Group 0

## 项目亮点

1. **完整的量化流程**: 从数据处理到模型预测的完整实现
2. **Alpha101因子**: 实现了经典的量化因子库
3. **模型集成**: 结合多个模型提升预测性能
4. **特征工程**: 丰富的特征生成和选择策略
5. **可扩展性**: 模块化设计，易于添加新因子和模型

## 进一步优化方向

### 1. 因子优化
- 实现更多Alpha因子（Alpha#27-100）
- 添加行业中性化处理
- 因子正交化处理

### 2. 模型优化
- 超参数调优（网格搜索、贝叶斯优化）
- 尝试深度学习模型（LSTM、Transformer）
- 时间序列交叉验证

### 3. 特征工程
- 添加宏观经济指标
- 市场情绪指标
- 技术形态识别

### 4. 风险管理
- 添加止损策略
- 仓位管理
- 回撤控制

## 常见问题

### Q1: 为什么测试集的y值都是0？
A: 这是项目设计，2020年1月1日后的数据y值为0，需要我们预测这些样本的y值。

### Q2: 如何提交预测结果？
A: 运行`main.py`后会生成`predictions.csv`文件，包含日期、股票ID和预测值。

### Q3: 如何添加新的Alpha因子？
A: 在`src/alpha101.py`中添加新的方法，格式为`alphaXXX()`，然后在`calculate_all_alphas()`中添加到计算列表。

### Q4: 模型训练时间过长怎么办？
A: 可以减少样本数量、减少特征数量或调整模型参数（如减少n_estimators）。

## 参考资料

1. [WorldQuant Alpha101 Paper](https://arxiv.org/abs/1601.00991)
2. [LightGBM Documentation](https://lightgbm.readthedocs.io/)
3. [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 作者和贡献

本项目为股票预测量化研究项目，欢迎提出改进建议和贡献代码。

## 许可证

MIT License

---

**注意**: 本项目仅用于学习和研究目的，不构成任何投资建议。实际投资需谨慎，风险自负。

