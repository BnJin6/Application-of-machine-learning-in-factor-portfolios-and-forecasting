# -*- coding: utf-8 -*-
"""
评估脚本 - 评估模型性能和因子有效性
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import spearmanr, pearsonr

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False


def calculate_ic(predictions, actuals):
    """
    计算信息系数 (IC)
    :param predictions: 预测值
    :param actuals: 实际值
    :return: IC值
    """
    ic_pearson, _ = pearsonr(predictions, actuals)
    ic_spearman, _ = spearmanr(predictions, actuals)
    
    return {
        'IC (Pearson)': ic_pearson,
        'IC (Spearman)': ic_spearman
    }


def calculate_rank_ic(predictions, actuals, n_groups=10):
    """
    计算分组收益
    :param predictions: 预测值
    :param actuals: 实际值
    :param n_groups: 分组数量
    :return: 各组平均收益
    """
    df = pd.DataFrame({
        'pred': predictions,
        'actual': actuals
    })
    
    # 根据预测值分组
    df['group'] = pd.qcut(df['pred'], n_groups, labels=False, duplicates='drop')
    
    # 计算各组平均收益
    group_returns = df.groupby('group')['actual'].mean()
    
    return group_returns


def plot_prediction_vs_actual(predictions, actuals, save_path='results/pred_vs_actual.png'):
    """
    绘制预测值vs实际值散点图
    :param predictions: 预测值
    :param actuals: 实际值
    :param save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(actuals, predictions, alpha=0.5, s=1)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('实际值', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.title('预测值 vs 实际值', fontsize=14)
    
    # 计算相关系数
    corr = np.corrcoef(actuals, predictions)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    plt.close()


def plot_group_returns(group_returns, save_path='results/group_returns.png'):
    """
    绘制分组收益图
    :param group_returns: 分组收益Series
    :param save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    group_returns.plot(kind='bar', color='steelblue')
    plt.xlabel('分组 (0=最低预测, 9=最高预测)', fontsize=12)
    plt.ylabel('平均收益', fontsize=12)
    plt.title('不同预测分组的平均收益', fontsize=14)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    plt.close()


def plot_feature_importance(feature_importance_df, top_n=20, save_path='results/feature_importance.png'):
    """
    绘制特征重要性图
    :param feature_importance_df: 特征重要性DataFrame
    :param top_n: 显示前N个特征
    :param save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('重要性', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.title(f'Top {top_n} 特征重要性', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    plt.close()


def evaluate_predictions(pred_file='predictions.csv', actual_file='daily_data.csv'):
    """
    评估预测结果
    :param pred_file: 预测结果文件
    :param actual_file: 实际数据文件
    """
    print("=" * 80)
    print("模型评估报告")
    print("=" * 80)
    
    # 加载预测结果
    predictions = pd.read_csv(pred_file)
    print(f"\n加载预测结果: {len(predictions)} 条")
    
    # 加载实际数据
    actuals = pd.read_csv(actual_file)
    actuals['date'] = pd.to_datetime(actuals['date'])
    
    # 合并预测和实际值
    merged = predictions.merge(
        actuals[['date', 'instrument_id', 'y']], 
        on=['date', 'instrument_id'],
        how='left'
    )
    
    # 只评估有实际值的样本
    eval_data = merged[merged['y'] != 0].copy()
    
    if len(eval_data) == 0:
        print("\n警告: 测试集没有实际的y值，无法进行评估")
        print("这是正常的，因为测试集(2020年后)的y值为0，需要提交预测结果")
        return
    
    print(f"可评估样本数: {len(eval_data)}")
    
    # 计算评估指标
    print("\n【评估指标】")
    print("-" * 80)
    
    # 相关系数
    correlation = np.corrcoef(eval_data['y'], eval_data['predicted_y'])[0, 1]
    print(f"Pearson相关系数: {correlation:.6f}")
    
    # IC指标
    ic_metrics = calculate_ic(eval_data['predicted_y'], eval_data['y'])
    for metric, value in ic_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # RMSE和MAE
    rmse = np.sqrt(np.mean((eval_data['y'] - eval_data['predicted_y']) ** 2))
    mae = np.mean(np.abs(eval_data['y'] - eval_data['predicted_y']))
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    # 分组收益分析
    print("\n【分组收益分析】")
    print("-" * 80)
    group_returns = calculate_rank_ic(eval_data['predicted_y'], eval_data['y'], n_groups=10)
    print(group_returns)
    
    # 多空收益
    long_short_return = group_returns.iloc[-1] - group_returns.iloc[0]
    print(f"\n多空收益 (最高组 - 最低组): {long_short_return:.6f}")
    
    # 绘图
    print("\n【生成可视化图表】")
    print("-" * 80)
    plot_prediction_vs_actual(eval_data['predicted_y'], eval_data['y'])
    plot_group_returns(group_returns)
    
    print("\n评估完成!")


def analyze_alpha_factors(data_file='daily_data.csv', alpha_list=None):
    """
    分析Alpha因子的有效性
    :param data_file: 数据文件
    :param alpha_list: 要分析的alpha列表
    """
    from src.data_loader import DataLoader
    from src.feature_generator import FeatureGenerator
    
    print("=" * 80)
    print("Alpha因子分析")
    print("=" * 80)
    
    # 加载数据
    data_loader = DataLoader(data_file)
    df = data_loader.load_data()
    df = data_loader.preprocess_data()
    
    # 只使用训练集数据
    train_data = df[df['date'] < '2020-01-01'].copy()
    
    # 生成Alpha因子
    feature_gen = FeatureGenerator(train_data)
    if alpha_list is None:
        alpha_list = list(range(1, 27)) + [101]
    
    features = feature_gen.generate_all_features(alpha_list=alpha_list)
    
    # 计算每个Alpha因子与目标的相关性
    alpha_cols = [col for col in features.columns if col.startswith('alpha')]
    
    correlations = {}
    for col in alpha_cols:
        valid_data = features[[col, 'y']].dropna()
        if len(valid_data) > 0 and valid_data['y'].std() > 0:
            corr = valid_data[col].corr(valid_data['y'])
            if not np.isnan(corr):
                correlations[col] = abs(corr)
    
    # 排序
    sorted_alphas = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("\nAlpha因子与目标变量的相关性:")
    print("-" * 80)
    for alpha, corr in sorted_alphas[:20]:
        print(f"{alpha}: {corr:.6f}")
    
    # 保存结果
    alpha_corr_df = pd.DataFrame(sorted_alphas, columns=['Alpha', 'Correlation'])
    alpha_corr_df.to_csv('results/alpha_correlations.csv', index=False)
    print(f"\nAlpha相关性已保存到: results/alpha_correlations.csv")


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    
    # 评估预测结果
    if os.path.exists('predictions.csv'):
        evaluate_predictions()
    else:
        print("未找到predictions.csv文件，请先运行main.py进行预测")
    
    # 分析Alpha因子（可选，耗时较长）
    # analyze_alpha_factors()

