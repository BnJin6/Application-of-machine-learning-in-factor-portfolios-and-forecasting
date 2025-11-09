# -*- coding: utf-8 -*-
"""
特征生成模块
整合Alpha因子和其他技术指标
"""
import pandas as pd
import numpy as np
from src.alpha101 import Alpha101
from src.data_loader import FeatureEngineer


class FeatureGenerator:
    """特征生成器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化特征生成器
        :param data: 原始数据
        """
        self.data = data.copy()
        self.features = None
        
    def generate_alpha_features(self, alpha_list=None):
        """
        生成Alpha101因子特征
        :param alpha_list: 要生成的alpha列表
        :return: 包含alpha因子的DataFrame
        """
        print("开始生成Alpha101因子...")
        alpha_calculator = Alpha101(self.data)
        alpha_dict = alpha_calculator.calculate_all_alphas(alpha_list)
        alpha_df = alpha_calculator.get_alpha_dataframe(alpha_dict)
        
        print(f"Alpha因子生成完成，共 {len(alpha_dict)} 个因子")
        return alpha_df
    
    def generate_technical_features(self):
        """
        生成技术指标特征
        :return: 包含技术指标的DataFrame
        """
        print("开始生成技术指标特征...")
        fe = FeatureEngineer(self.data)
        df_with_basic = fe.create_basic_features()
        df_with_market = fe.create_market_features()
        
        # 合并特征
        feature_df = df_with_basic.merge(
            df_with_market[['date', 'instrument_id', 'market_cap_rank', 'volume_rank', 'turnover_rate_rank']],
            on=['date', 'instrument_id'],
            how='left'
        )
        
        print("技术指标特征生成完成")
        return feature_df
    
    def generate_all_features(self, alpha_list=None):
        """
        生成所有特征
        :param alpha_list: 要生成的alpha列表
        :return: 包含所有特征的DataFrame
        """
        # 生成技术指标特征
        tech_features = self.generate_technical_features()
        
        # 生成Alpha因子
        alpha_features = self.generate_alpha_features(alpha_list)
        
        # 合并所有特征
        if not alpha_features.empty:
            all_features = tech_features.merge(
                alpha_features,
                on=['date', 'instrument_id'],
                how='left'
            )
        else:
            all_features = tech_features
        
        self.features = all_features
        print(f"所有特征生成完成，共 {len(all_features.columns)} 列")
        return all_features
    
    def select_features(self, feature_list=None):
        """
        选择特征子集
        :param feature_list: 特征名称列表，如果为None则返回所有特征
        :return: 选择后的特征DataFrame
        """
        if self.features is None:
            raise ValueError("请先调用 generate_all_features() 生成特征")
        
        if feature_list is None:
            # 排除非特征列
            exclude_cols = ['instrument_id', 'date', 'y', 'next_open', 'adjustment', 'type']
            feature_list = [col for col in self.features.columns if col not in exclude_cols]
        
        # 确保必要的列存在
        required_cols = ['date', 'instrument_id', 'y']
        selected_cols = required_cols + [col for col in feature_list if col in self.features.columns]
        
        return self.features[selected_cols]
    
    def get_feature_importance_by_correlation(self, top_n=50):
        """
        根据与目标变量的相关性选择特征
        :param top_n: 选择前N个特征
        :return: 特征重要性DataFrame
        """
        if self.features is None:
            raise ValueError("请先调用 generate_all_features() 生成特征")
        
        # 计算与目标变量的相关性
        exclude_cols = ['instrument_id', 'date', 'y', 'next_open']
        feature_cols = [col for col in self.features.columns if col not in exclude_cols]
        
        correlations = {}
        for col in feature_cols:
            try:
                corr = self.features[col].corr(self.features['y'])
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            except:
                continue
        
        # 排序
        importance_df = pd.DataFrame(list(correlations.items()), 
                                    columns=['feature', 'correlation'])
        importance_df = importance_df.sort_values('correlation', ascending=False)
        
        print(f"\n特征重要性（Top {top_n}）:")
        print(importance_df.head(top_n))
        
        return importance_df.head(top_n)


class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, X, y):
        """
        初始化特征选择器
        :param X: 特征DataFrame
        :param y: 目标变量
        """
        self.X = X
        self.y = y
        
    def remove_low_variance_features(self, threshold=0.01):
        """
        移除低方差特征
        :param threshold: 方差阈值
        :return: 选择后的特征列表
        """
        variances = self.X.var()
        selected_features = variances[variances > threshold].index.tolist()
        print(f"移除低方差特征后，剩余 {len(selected_features)} 个特征")
        return selected_features
    
    def remove_high_correlation_features(self, threshold=0.95):
        """
        移除高相关性特征
        :param threshold: 相关性阈值
        :return: 选择后的特征列表
        """
        corr_matrix = self.X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        selected_features = [col for col in self.X.columns if col not in to_drop]
        print(f"移除高相关性特征后，剩余 {len(selected_features)} 个特征")
        return selected_features
    
    def select_by_importance(self, method='correlation', top_n=50):
        """
        根据重要性选择特征（优化版）
        :param method: 选择方法 ('correlation', 'mutual_info')
        :param top_n: 选择前N个特征
        :return: 选择后的特征列表
        """
        if method == 'correlation':
            correlations = {}
            for col in self.X.columns:
                try:
                    corr = self.X[col].corr(self.y)
                    if not np.isnan(corr) and abs(corr) > 0.001:  # 过滤极低相关性
                        correlations[col] = abs(corr)
                except:
                    continue
            
            # 如果相关特征太少，降低阈值
            if len(correlations) < top_n:
                correlations = {}
                for col in self.X.columns:
                    try:
                        corr = self.X[col].corr(self.y)
                        if not np.isnan(corr):
                            correlations[col] = abs(corr)
                    except:
                        continue
            
            sorted_features = sorted(correlations.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)
            selected_features = [f[0] for f in sorted_features[:top_n]]
            
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            
            # 处理缺失值
            X_filled = self.X.fillna(0)
            mi_scores = mutual_info_regression(X_filled, self.y)
            mi_dict = dict(zip(self.X.columns, mi_scores))
            sorted_features = sorted(mi_dict.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)
            selected_features = [f[0] for f in sorted_features[:top_n]]
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        print(f"根据{method}选择了 {len(selected_features)} 个特征")
        return selected_features

