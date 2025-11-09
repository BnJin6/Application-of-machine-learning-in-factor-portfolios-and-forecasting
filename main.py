# -*- coding: utf-8 -*-
"""
主程序 - 股票预测项目
完整的训练和预测流程
"""
import pandas as pd
import numpy as np
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader, prepare_data_for_model
from src.feature_generator import FeatureGenerator, FeatureSelector
from src.model import StockPredictor, ModelEnsemble, cross_validate_model
from src.logger import create_logger


def main():
    """主函数"""
    
    # 创建日志记录器
    logger = create_logger('stock_prediction', 'logs')
    
    logger.log_section("股票预测项目 - Alpha101因子量化预测")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 记录运行配置
    config = {
        '数据文件': 'daily_data.csv',
        '训练集截止日期': '2019-12-31',
        '测试集起始日期': '2020-01-01',
        'Alpha因子数量': 27,
        '模型类型': ['LightGBM', 'XGBoost'],
        '特征选择数量': 50
    }
    logger.log_dict(config, "运行配置:")
    
    # ==================== 1. 数据加载 ====================
    logger.log_section("【步骤1】数据加载与预处理", char='-')
    
    try:
        data_loader = DataLoader('daily_data.csv')
        df = data_loader.load_data()
        logger.log_dataframe_info(df, "原始数据")
        
        df = data_loader.preprocess_data()
        logger.log_dataframe_info(df, "预处理后数据")
        
        # 划分训练集和测试集
        train_data, test_data = data_loader.split_data(df)
        logger.info(f"训练集样本数: {len(train_data)}")
        logger.info(f"测试集样本数: {len(test_data)}")
        logger.info(f"训练集股票数: {train_data['instrument_id'].nunique()}")
        logger.info(f"测试集股票数: {test_data['instrument_id'].nunique()}")
        
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise
    
    # ==================== 2. 特征生成 ====================
    logger.log_section("【步骤2】特征工程", char='-')
    
    try:
        # 生成训练集特征
        logger.info("开始生成训练集特征...")
        train_feature_gen = FeatureGenerator(train_data)
        
        # 选择要计算的alpha因子（这里选择前26个和alpha101）
        alpha_list = list(range(1, 27)) + [101]
        logger.info(f"计算Alpha因子: {alpha_list}")
        
        train_features = train_feature_gen.generate_all_features(alpha_list=alpha_list)
        logger.log_dataframe_info(train_features, "训练集特征")
        
        # 生成测试集特征
        logger.info("开始生成测试集特征...")
        test_feature_gen = FeatureGenerator(test_data)
        test_features = test_feature_gen.generate_all_features(alpha_list=alpha_list)
        logger.log_dataframe_info(test_features, "测试集特征")
        
    except Exception as e:
        logger.error(f"特征生成失败: {str(e)}")
        raise
    
    # ==================== 3. 特征选择 ====================
    logger.log_section("【步骤3】特征选择", char='-')
    
    try:
        # 移除包含目标变量为0的测试集样本（这些是需要预测的）
        train_clean = train_features[train_features['y'] != 0].copy()
        logger.info(f"移除y=0样本后，训练集样本数: {len(train_clean)}")
        
        # 准备特征和目标变量
        exclude_cols = ['instrument_id', 'date', 'y', 'next_open', 'adjustment', 'type', 
                       'adj_factor', 'a_share_capital', 'total_capital', 'float_a_share_capital',
                       'turnover', 'turnover_rate']
        feature_cols = [col for col in train_clean.columns if col not in exclude_cols]
        logger.info(f"初始特征数量: {len(feature_cols)}")
        
        # 移除包含过多缺失值的特征
        missing_ratio = train_clean[feature_cols].isnull().sum() / len(train_clean)
        valid_features = missing_ratio[missing_ratio < 0.5].index.tolist()
        logger.info(f"移除高缺失率特征后，剩余 {len(valid_features)} 个特征")
        
        # 填充缺失值
        train_clean[valid_features] = train_clean[valid_features].fillna(0)
        
        # 准备训练数据
        X_train, y_train = prepare_data_for_model(train_clean, valid_features, 'y')
        logger.info(f"训练集大小: {X_train.shape}")
        logger.info(f"目标变量统计: 均值={y_train.mean():.6f}, 标准差={y_train.std():.6f}")
        
        # 特征选择
        logger.info("开始特征选择...")
        selector = FeatureSelector(X_train, y_train)
        
        # 移除低方差特征
        selected_features = selector.remove_low_variance_features(threshold=0.001)
        logger.info(f"移除低方差特征后: {len(selected_features)} 个特征")
        
        # 移除高相关性特征
        X_train_selected = X_train[selected_features]
        selector_corr = FeatureSelector(X_train_selected, y_train)
        final_features = selector_corr.remove_high_correlation_features(threshold=0.95)
        logger.info(f"移除高相关性特征后: {len(final_features)} 个特征")
        
        # 根据相关性选择Top特征（增加特征数量）
        X_train_final = X_train[final_features]
        selector_final = FeatureSelector(X_train_final, y_train)
        
        # 尝试使用更多特征
        n_features = min(len(final_features), 45)  # 使用更多特征
        top_features = selector_final.select_by_importance(method='correlation', top_n=n_features)
        
        X_train_top = X_train[top_features]
        logger.info(f"最终选择 {len(top_features)} 个特征用于建模")
        logger.info(f"Top 10特征: {top_features[:10]}")
        
        # 记录特征与目标的相关性
        feature_correlations = {}
        for feat in top_features:
            corr = X_train_top[feat].corr(y_train)
            if not np.isnan(corr):
                feature_correlations[feat] = abs(corr)
        
        sorted_corrs = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"\nTop 5 特征相关性:")
        for feat, corr in sorted_corrs[:5]:
            logger.info(f"  {feat}: {corr:.6f}")
        
    except Exception as e:
        logger.error(f"特征选择失败: {str(e)}")
        raise
    
    # ==================== 4. 模型训练 ====================
    logger.log_section("【步骤4】模型训练", char='-')
    
    try:
        # 划分训练集和验证集（使用更多训练数据）
        split_idx = int(len(X_train_top) * 0.85)  # 使用85%作为训练集
        X_tr, X_val = X_train_top.iloc[:split_idx], X_train_top.iloc[split_idx:]
        y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
        
        logger.info(f"训练集: {X_tr.shape}, 验证集: {X_val.shape}")
        logger.info(f"训练集目标变量统计: 均值={y_tr.mean():.6f}, 标准差={y_tr.std():.6f}")
        logger.info(f"验证集目标变量统计: 均值={y_val.mean():.6f}, 标准差={y_val.std():.6f}")
        
        # 训练LightGBM模型
        logger.info("=" * 40)
        logger.info("训练LightGBM模型...")
        logger.info("=" * 40)
        lgb_model = StockPredictor(model_type='lightgbm')
        lgb_model.train(X_tr, y_tr, X_val, y_val)
        
        # 评估模型
        logger.info("\nLightGBM验证集评估:")
        lgb_metrics = lgb_model.evaluate(X_val, y_val)
        logger.log_dict(lgb_metrics, "LightGBM性能指标:")
        
        # 显示特征重要性
        lgb_importance = lgb_model.get_feature_importance(top_n=20)
        if lgb_importance is not None:
            logger.info("\nLightGBM Top 10特征重要性:")
            for idx, row in lgb_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # 训练XGBoost模型
        logger.info("\n" + "=" * 40)
        logger.info("训练XGBoost模型...")
        logger.info("=" * 40)
        xgb_model = StockPredictor(model_type='xgboost')
        xgb_model.train(X_tr, y_tr, X_val, y_val)
        
        # 评估模型
        logger.info("\nXGBoost验证集评估:")
        xgb_metrics = xgb_model.evaluate(X_val, y_val)
        logger.log_dict(xgb_metrics, "XGBoost性能指标:")
        
        # 显示特征重要性
        xgb_importance = xgb_model.get_feature_importance(top_n=20)
        if xgb_importance is not None:
            logger.info("\nXGBoost Top 10特征重要性:")
            for idx, row in xgb_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        raise
    
    # ==================== 5. 模型集成 ====================
    logger.log_section("【步骤5】模型集成", char='-')
    
    try:
        ensemble = ModelEnsemble()
        ensemble.add_model(lgb_model)
        ensemble.add_model(xgb_model)
        logger.info("已添加2个模型到集成器")
        
        logger.info("\n集成模型验证集评估:")
        ensemble_metrics = ensemble.evaluate(X_val, y_val, method='average')
        logger.log_dict(ensemble_metrics, "集成模型性能指标:")
        
    except Exception as e:
        logger.error(f"模型集成失败: {str(e)}")
        raise
    
    # ==================== 6. 测试集预测 ====================
    logger.log_section("【步骤6】测试集预测", char='-')
    
    try:
        # 准备测试集数据
        test_clean = test_features.copy()
        test_clean[valid_features] = test_clean[valid_features].fillna(0)
        
        # 确保测试集有相同的特征
        X_test = test_clean[top_features].fillna(0)
        
        logger.info(f"测试集大小: {X_test.shape}")
        logger.info(f"测试集日期范围: {test_clean['date'].min()} 至 {test_clean['date'].max()}")
        
        # 使用集成模型预测
        logger.info("开始预测...")
        test_predictions = ensemble.predict(X_test, method='average')
        
        logger.info(f"预测值统计: 均值={test_predictions.mean():.6f}, 标准差={test_predictions.std():.6f}")
        logger.info(f"预测值范围: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")
        
        # 保存预测结果
        result_df = test_clean[['date', 'instrument_id']].copy()
        result_df['predicted_y'] = test_predictions
        
        # 保存到CSV
        result_df.to_csv('predictions.csv', index=False)
        logger.info(f"预测结果已保存到: predictions.csv")
        logger.info(f"预测样本数: {len(result_df)}")
        
        # 保存预测结果的详细统计
        pred_stats = {
            '预测样本数': len(result_df),
            '预测股票数': result_df['instrument_id'].nunique(),
            '预测日期数': result_df['date'].nunique(),
            '预测值均值': float(test_predictions.mean()),
            '预测值标准差': float(test_predictions.std()),
            '预测值最小值': float(test_predictions.min()),
            '预测值最大值': float(test_predictions.max())
        }
        logger.log_dict(pred_stats, "预测结果统计:")
        
    except Exception as e:
        logger.error(f"测试集预测失败: {str(e)}")
        raise
    
    # ==================== 7. 保存模型 ====================
    logger.log_section("【步骤7】保存模型", char='-')
    
    try:
        lgb_model.save_model('models/lgb_model.pkl')
        logger.info("LightGBM模型已保存到: models/lgb_model.pkl")
        
        xgb_model.save_model('models/xgb_model.pkl')
        logger.info("XGBoost模型已保存到: models/xgb_model.pkl")
        
        # 保存特征列表
        import joblib
        joblib.dump(top_features, 'models/feature_list.pkl')
        logger.info("特征列表已保存到: models/feature_list.pkl")
        
        # 保存运行配置和结果
        run_summary = {
            '运行时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '数据集': {
                '训练集样本数': len(train_data),
                '测试集样本数': len(test_data),
                '特征数量': len(top_features)
            },
            '模型性能': {
                'LightGBM': lgb_metrics,
                'XGBoost': xgb_metrics,
                'Ensemble': ensemble_metrics
            },
            '预测结果': pred_stats,
            '特征列表': top_features
        }
        
        with open('logs/run_summary.json', 'w', encoding='utf-8') as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2, default=str)
        logger.info("运行摘要已保存到: logs/run_summary.json")
        
    except Exception as e:
        logger.error(f"保存模型失败: {str(e)}")
        raise
    
    # ==================== 8. 结果总结 ====================
    logger.log_section("【步骤8】结果总结")
    
    logger.info("\n模型性能对比:")
    logger.info(f"  LightGBM Correlation: {lgb_metrics['Correlation']:.6f}")
    logger.info(f"  XGBoost Correlation: {xgb_metrics['Correlation']:.6f}")
    logger.info(f"  Ensemble Correlation: {ensemble_metrics['Correlation']:.6f}")
    
    logger.info("\n最佳模型: " + ("Ensemble" if ensemble_metrics['Correlation'] >= max(lgb_metrics['Correlation'], xgb_metrics['Correlation']) else 
                                 ("LightGBM" if lgb_metrics['Correlation'] > xgb_metrics['Correlation'] else "XGBoost")))
    
    logger.info(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log_section("项目完成!")
    logger.info(f"\n详细日志已保存到: {logger.get_log_file()}")


if __name__ == '__main__':
    # 创建必要的目录
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    try:
        main()
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 尝试记录错误到日志
        try:
            from src.logger import create_logger
            error_logger = create_logger('error', 'logs')
            error_logger.error(f"程序执行出错: {str(e)}")
            error_logger.error(traceback.format_exc())
        except:
            pass

