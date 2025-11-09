# -*- coding: utf-8 -*-
"""
模型训练和预测模块
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb


class StockPredictor:
    """股票预测模型"""
    
    def __init__(self, model_type='lightgbm', params=None):
        """
        初始化预测模型
        :param model_type: 模型类型 ('lightgbm', 'xgboost', 'linear')
        :param params: 模型参数
        """
        self.model_type = model_type
        self.params = params or self._get_default_params()
        self.model = None
        self.feature_importance = None
        
    def _get_default_params(self):
        """获取默认参数（优化版）"""
        if self.model_type == 'lightgbm':
            return {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 63,  # 增加叶子节点
                'max_depth': 8,  # 增加树深度
                'learning_rate': 0.03,  # 降低学习率
                'feature_fraction': 0.7,  # 减少特征采样，增加多样性
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'min_child_samples': 50,  # 增加最小样本数
                'reg_alpha': 0.1,  # L1正则化
                'reg_lambda': 0.1,  # L2正则化
                'verbose': -1,
                'n_estimators': 2000,  # 增加树的数量
                'early_stopping_rounds': 100
            }
        elif self.model_type == 'xgboost':
            return {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 8,  # 增加深度
                'learning_rate': 0.03,  # 降低学习率
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'n_estimators': 2000,
                'early_stopping_rounds': 100
            }
        else:
            return {}
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练模型
        :param X_train: 训练集特征
        :param y_train: 训练集目标
        :param X_val: 验证集特征
        :param y_val: 验证集目标
        :return: 训练好的模型
        """
        print(f"开始训练 {self.model_type} 模型...")
        print(f"训练集大小: {X_train.shape}")
        
        if self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**self.params)
            
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(self.params.get('early_stopping_rounds', 50))]
                )
            else:
                self.model.fit(X_train, y_train)
            
            # 获取特征重要性
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**self.params)
            
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
            
            # 获取特征重要性
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        elif self.model_type == 'linear':
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_train, y_train)
            
            # 获取特征重要性（系数绝对值）
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(self.model.coef_)
            }).sort_values('importance', ascending=False)
        
        print("模型训练完成")
        return self.model
    
    def predict(self, X):
        """
        预测
        :param X: 特征数据
        :return: 预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train() 方法")
        
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        评估模型
        :param X: 特征数据
        :param y: 真实目标值
        :return: 评估指标字典
        """
        y_pred = self.predict(X)
        
        # 计算各种指标
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        correlation = np.corrcoef(y, y_pred)[0, 1]
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Correlation': correlation
        }
        
        print("\n模型评估结果:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        
        return metrics
    
    def save_model(self, filepath):
        """
        保存模型
        :param filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("模型未训练，无法保存")
        
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_importance': self.feature_importance
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        :param filepath: 模型文件路径
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.params = model_data['params']
        self.feature_importance = model_data['feature_importance']
        print(f"模型已从 {filepath} 加载")
    
    def get_feature_importance(self, top_n=20):
        """
        获取特征重要性
        :param top_n: 显示前N个特征
        :return: 特征重要性DataFrame
        """
        if self.feature_importance is None:
            print("特征重要性未计算")
            return None
        
        print(f"\n特征重要性 (Top {top_n}):")
        print(self.feature_importance.head(top_n))
        return self.feature_importance.head(top_n)


class ModelEnsemble:
    """模型集成"""
    
    def __init__(self, models=None):
        """
        初始化模型集成
        :param models: 模型列表
        """
        self.models = models or []
        self.weights = None
        
    def add_model(self, model):
        """
        添加模型
        :param model: StockPredictor实例
        """
        self.models.append(model)
    
    def train_all(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练所有模型
        :param X_train: 训练集特征
        :param y_train: 训练集目标
        :param X_val: 验证集特征
        :param y_val: 验证集目标
        """
        for i, model in enumerate(self.models):
            print(f"\n训练模型 {i+1}/{len(self.models)}")
            model.train(X_train, y_train, X_val, y_val)
    
    def predict(self, X, method='average'):
        """
        集成预测
        :param X: 特征数据
        :param method: 集成方法 ('average', 'weighted')
        :return: 预测结果
        """
        predictions = np.array([model.predict(X) for model in self.models])
        
        if method == 'average':
            return predictions.mean(axis=0)
        elif method == 'weighted':
            if self.weights is None:
                self.weights = np.ones(len(self.models)) / len(self.models)
            return np.average(predictions, axis=0, weights=self.weights)
        else:
            raise ValueError(f"不支持的集成方法: {method}")
    
    def evaluate(self, X, y, method='average'):
        """
        评估集成模型
        :param X: 特征数据
        :param y: 真实目标值
        :param method: 集成方法
        :return: 评估指标字典
        """
        y_pred = self.predict(X, method)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        correlation = np.corrcoef(y, y_pred)[0, 1]
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Correlation': correlation
        }
        
        print(f"\n集成模型评估结果 ({method}):")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        
        return metrics


def cross_validate_model(X, y, model_type='lightgbm', n_splits=5, params=None):
    """
    时间序列交叉验证
    :param X: 特征数据
    :param y: 目标变量
    :param model_type: 模型类型
    :param n_splits: 折数
    :param params: 模型参数
    :return: 交叉验证结果
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\n===== Fold {fold}/{n_splits} =====")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练模型
        model = StockPredictor(model_type=model_type, params=params)
        model.train(X_train, y_train, X_val, y_val)
        
        # 评估
        metrics = model.evaluate(X_val, y_val)
        cv_scores.append(metrics)
    
    # 计算平均分数
    avg_scores = {}
    for metric in cv_scores[0].keys():
        avg_scores[metric] = np.mean([score[metric] for score in cv_scores])
    
    print("\n===== 交叉验证平均结果 =====")
    for metric, value in avg_scores.items():
        print(f"{metric}: {value:.6f}")
    
    return cv_scores, avg_scores

