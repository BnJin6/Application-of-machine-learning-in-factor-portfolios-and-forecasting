# 日志系统使用指南

## 概述

本项目实现了完整的日志记录系统，记录训练过程、预测结果和所有关键步骤的详细信息。

## 日志文件位置

所有日志文件保存在 `logs/` 目录下：

```
logs/
├── stock_prediction_20241105_143025.log    # 主程序运行日志（带时间戳）
├── error_20241105_143530.log               # 错误日志（如有）
└── run_summary.json                        # 运行摘要（JSON格式）
```

## 日志内容

### 1. 主程序日志 (`stock_prediction_*.log`)

记录完整的训练和预测过程，包括：

#### 运行配置
```
运行配置:
  数据文件: daily_data.csv
  训练集截止日期: 2019-12-31
  测试集起始日期: 2020-01-01
  Alpha因子数量: 27
  模型类型: ['LightGBM', 'XGBoost']
  特征选择数量: 50
```

#### 数据加载信息
```
【步骤1】数据加载与预处理
原始数据 信息:
  形状: (953751, 16)
  列数: 16
  行数: 953751
  内存使用: 116.45 MB
训练集样本数: 850000
测试集样本数: 103751
训练集股票数: 3500
测试集股票数: 3500
```

#### 特征工程信息
```
【步骤2】特征工程
开始生成训练集特征...
计算Alpha因子: [1, 2, 3, ..., 26, 101]
正在计算 alpha001...
正在计算 alpha002...
...
训练集特征 信息:
  形状: (850000, 45)
  列数: 45
  行数: 850000
```

#### 特征选择过程
```
【步骤3】特征选择
移除y=0样本后，训练集样本数: 820000
初始特征数量: 42
移除高缺失率特征后，剩余 38 个特征
训练集大小: (820000, 38)
目标变量统计: 均值=0.000123, 标准差=0.015678
开始特征选择...
移除低方差特征后: 35 个特征
移除高相关性特征后: 30 个特征
最终选择 50 个特征用于建模
Top 10特征: ['alpha101', 'alpha001', 'momentum_20', ...]
```

#### 模型训练详情
```
【步骤4】模型训练
训练集: (656000, 50), 验证集: (164000, 50)

========================================
训练LightGBM模型...
========================================
开始训练 lightgbm 模型...
训练集大小: (656000, 50)
模型训练完成

LightGBM验证集评估:
模型评估结果:
MSE: 0.000245
RMSE: 0.015656
MAE: 0.010234
Correlation: 0.085432

LightGBM性能指标:
  MSE: 0.000245
  RMSE: 0.015656
  MAE: 0.010234
  Correlation: 0.085432

LightGBM Top 10特征重要性:
  alpha101: 1234.5678
  alpha001: 987.6543
  momentum_20: 765.4321
  ...
```

#### 模型集成
```
【步骤5】模型集成
已添加2个模型到集成器

集成模型验证集评估:
集成模型性能指标:
  MSE: 0.000240
  RMSE: 0.015492
  MAE: 0.010123
  Correlation: 0.087654
```

#### 预测结果
```
【步骤6】测试集预测
测试集大小: (103751, 50)
测试集日期范围: 2020-01-01 至 2024-12-31
开始预测...
预测值统计: 均值=0.000156, 标准差=0.012345
预测值范围: [-0.045678, 0.056789]
预测结果已保存到: predictions.csv
预测样本数: 103751

预测结果统计:
  预测样本数: 103751
  预测股票数: 3500
  预测日期数: 1200
  预测值均值: 0.000156
  预测值标准差: 0.012345
  预测值最小值: -0.045678
  预测值最大值: 0.056789
```

#### 结果总结
```
【步骤8】结果总结
模型性能对比:
  LightGBM Correlation: 0.085432
  XGBoost Correlation: 0.084567
  Ensemble Correlation: 0.087654

最佳模型: Ensemble

结束时间: 2024-11-05 14:35:30
详细日志已保存到: logs/stock_prediction_20241105_143025.log
```

### 2. 运行摘要 (`run_summary.json`)

JSON格式的结构化运行结果，方便程序读取和分析：

```json
{
  "运行时间": "2024-11-05 14:35:30",
  "数据集": {
    "训练集样本数": 850000,
    "测试集样本数": 103751,
    "特征数量": 50
  },
  "模型性能": {
    "LightGBM": {
      "MSE": 0.000245,
      "RMSE": 0.015656,
      "MAE": 0.010234,
      "Correlation": 0.085432
    },
    "XGBoost": {
      "MSE": 0.000248,
      "RMSE": 0.015748,
      "MAE": 0.010345,
      "Correlation": 0.084567
    },
    "Ensemble": {
      "MSE": 0.000240,
      "RMSE": 0.015492,
      "MAE": 0.010123,
      "Correlation": 0.087654
    }
  },
  "预测结果": {
    "预测样本数": 103751,
    "预测股票数": 3500,
    "预测日期数": 1200,
    "预测值均值": 0.000156,
    "预测值标准差": 0.012345,
    "预测值最小值": -0.045678,
    "预测值最大值": 0.056789
  },
  "特征列表": [
    "alpha101",
    "alpha001",
    "momentum_20",
    "..."
  ]
}
```

### 3. 错误日志 (`error_*.log`)

如果程序执行出错，会自动创建错误日志文件，记录：
- 错误信息
- 完整的堆栈跟踪
- 错误发生的时间和位置

## 日志级别

日志系统支持多个级别：

- **DEBUG**: 详细的调试信息（仅记录到文件）
- **INFO**: 一般信息（记录到文件和控制台）
- **WARNING**: 警告信息
- **ERROR**: 错误信息
- **CRITICAL**: 严重错误

## 如何使用日志

### 在代码中使用日志

```python
from src.logger import create_logger

# 创建日志记录器
logger = create_logger('my_module', 'logs')

# 记录不同级别的日志
logger.info("这是一条信息")
logger.debug("这是调试信息")
logger.warning("这是警告")
logger.error("这是错误")

# 记录章节标题
logger.log_section("数据处理")

# 记录字典数据
config = {'param1': 'value1', 'param2': 'value2'}
logger.log_dict(config, "配置信息:")

# 记录DataFrame信息
logger.log_dataframe_info(df, "数据集")
```

### 查看日志

#### 1. 实时查看日志（Linux/Mac）
```bash
tail -f logs/stock_prediction_*.log
```

#### 2. 查看完整日志
```bash
cat logs/stock_prediction_20241105_143025.log
```

#### 3. 搜索特征日志内容
```bash
grep "Correlation" logs/stock_prediction_*.log
grep "ERROR" logs/stock_prediction_*.log
```

#### 4. 读取JSON摘要
```python
import json

with open('logs/run_summary.json', 'r', encoding='utf-8') as f:
    summary = json.load(f)
    
print(f"最佳相关系数: {summary['模型性能']['Ensemble']['Correlation']}")
```

## 日志文件管理

### 自动命名

日志文件名自动包含时间戳，格式为：
```
{name}_{YYYYMMDD}_{HHMMSS}.log
```

例如：`stock_prediction_20241105_143025.log`

### 清理旧日志

可以定期清理旧的日志文件：

```bash
# 删除7天前的日志
find logs/ -name "*.log" -mtime +7 -delete

# 只保留最近的10个日志文件
ls -t logs/*.log | tail -n +11 | xargs rm
```

### 日志归档

对于重要的运行结果，建议归档保存：

```bash
# 创建归档目录
mkdir -p logs/archive

# 归档特定日期的日志
mv logs/stock_prediction_20241105_*.log logs/archive/
mv logs/run_summary.json logs/archive/run_summary_20241105.json
```

## 日志分析

### 提取关键指标

```python
import re

def extract_metrics(log_file):
    """从日志文件中提取关键指标"""
    metrics = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 提取相关系数
        corr_pattern = r'Correlation: ([\d.]+)'
        correlations = re.findall(corr_pattern, content)
        
        if len(correlations) >= 3:
            metrics['LightGBM_Corr'] = float(correlations[0])
            metrics['XGBoost_Corr'] = float(correlations[1])
            metrics['Ensemble_Corr'] = float(correlations[2])
    
    return metrics

# 使用示例
metrics = extract_metrics('logs/stock_prediction_20241105_143025.log')
print(metrics)
```

### 批量分析多次运行

```python
import json
import glob
import pandas as pd

def analyze_multiple_runs():
    """分析多次运行的结果"""
    results = []
    
    for json_file in glob.glob('logs/archive/run_summary_*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append({
                '运行时间': data['运行时间'],
                'LightGBM': data['模型性能']['LightGBM']['Correlation'],
                'XGBoost': data['模型性能']['XGBoost']['Correlation'],
                'Ensemble': data['模型性能']['Ensemble']['Correlation']
            })
    
    df = pd.DataFrame(results)
    print(df)
    print(f"\n平均Ensemble相关系数: {df['Ensemble'].mean():.6f}")
    
    return df

# 使用示例
df_results = analyze_multiple_runs()
```

## 最佳实践

1. **每次重要运行都保存日志**: 便于后续分析和对比
2. **定期归档日志**: 避免日志目录过大
3. **使用JSON摘要**: 方便程序化分析结果
4. **记录实验配置**: 在日志中记录所有超参数和配置
5. **版本控制**: 将重要的日志文件纳入版本控制

## 故障排查

### 问题1: 日志文件未生成

**解决方案**:
- 检查 `logs/` 目录是否存在
- 检查目录写入权限
- 查看控制台是否有错误信息

### 问题2: 日志内容不完整

**解决方案**:
- 确保程序正常结束（没有被强制终止）
- 检查磁盘空间是否充足
- 查看是否有异常退出

### 问题3: 无法读取JSON摘要

**解决方案**:
- 确保程序完整运行到最后
- 检查JSON文件格式是否正确
- 使用文本编辑器打开查看内容

## 总结

本日志系统提供了：
- ✅ 完整的训练过程记录
- ✅ 详细的性能指标
- ✅ 结构化的运行摘要
- ✅ 自动错误追踪
- ✅ 便于分析的格式

通过查看日志，你可以：
- 了解模型训练的每个步骤
- 对比不同模型的性能
- 追踪预测结果的统计信息
- 排查程序运行中的问题
- 分析多次实验的结果

