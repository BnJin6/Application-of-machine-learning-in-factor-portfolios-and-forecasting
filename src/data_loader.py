# -*- coding: utf-8 -*-
"""
数据加载和预处理模块
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_path: str):
        """
        初始化数据加载器
        :param data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        加载CSV数据
        :return: 加载的DataFrame
        """
        print(f"正在加载数据: {self.data_path}")
        
        # 尝试多种方式加载数据
        # 1. 尝试直接加载（可能是gzip压缩）
        try:
            print("尝试直接加载...")
            self.data = pd.read_csv(
                self.data_path, 
                compression='infer',  # 自动检测压缩格式
                encoding='utf-8',
                on_bad_lines='skip',
                low_memory=False
            )
            print(f"数据加载完成，共 {len(self.data)} 条记录")
            return self.data
        except Exception as e1:
            print(f"直接加载失败: {str(e1)[:100]}")
            
            # 2. 尝试使用latin1编码
            try:
                print("尝试使用latin1编码...")
                self.data = pd.read_csv(
                    self.data_path, 
                    encoding='latin1',
                    on_bad_lines='skip',
                    low_memory=False
                )
                print(f"数据加载完成，共 {len(self.data)} 条记录（latin1编码）")
                return self.data
            except Exception as e2:
                print(f"latin1编码失败: {str(e2)[:100]}")
                
                # 3. 尝试使用encoding_errors='ignore'
                try:
                    print("尝试忽略编码错误...")
                    self.data = pd.read_csv(
                        self.data_path, 
                        encoding='utf-8',
                        encoding_errors='ignore',
                        on_bad_lines='skip',
                        low_memory=False
                    )
                    print(f"数据加载完成，共 {len(self.data)} 条记录（忽略错误）")
                    return self.data
                except Exception as e3:
                    raise Exception(f"无法加载数据文件，所有方法都失败: {str(e3)}")
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        数据预处理
        :return: 预处理后的DataFrame
        """
        if self.data is None:
            self.load_data()
        
        df = self.data.copy()
        
        # 转换日期格式（使用errors='coerce'将无效日期转为NaT）
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # 移除日期无效的行
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            print(f"警告: 发现 {invalid_dates} 行日期无效，将被移除")
            df = df.dropna(subset=['date'])
        
        # 按股票和日期排序
        df = df.sort_values(['instrument_id', 'date']).reset_index(drop=True)
        
        # 转换数值列的数据类型
        numeric_cols = ['y', 'high', 'low', 'open', 'close', 'volume', 'vwap', 
                       'adjustment', 'type', 'a_share_capital', 'total_capital', 
                       'float_a_share_capital', 'turnover', 'turnover_rate', 'next_open']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 计算returns（收益率）- 使用对数收益率更稳定
        df['returns'] = df.groupby('instrument_id')['close'].apply(
            lambda x: np.log(x / x.shift(1))
        ).reset_index(level=0, drop=True)
        
        # 处理adjustment（复权因子）
        # 计算真实的价格变化率时需要考虑adjustment
        df['adj_factor'] = 1 + df['adjustment']
        
        # 处理缺失值（兼容pandas 2.x）
        df = df.ffill().bfill()
        
        print(f"数据预处理完成")
        return df
    
    def split_data(self, df: pd.DataFrame, 
                   train_end_date: str = '2019-12-31',
                   test_start_date: str = '2020-01-01') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        划分训练集和测试集
        :param df: 完整数据集
        :param train_end_date: 训练集结束日期
        :param test_start_date: 测试集开始日期
        :return: (训练集, 测试集)
        """
        train_data = df[df['date'] < train_end_date].copy()
        test_data = df[df['date'] >= test_start_date].copy()
        
        print(f"训练集: {len(train_data)} 条记录 (截止 {train_end_date})")
        print(f"测试集: {len(test_data)} 条记录 (从 {test_start_date} 开始)")
        
        return train_data, test_data
    
    def get_stock_panel(self, df: pd.DataFrame, stock_id: Optional[str] = None) -> pd.DataFrame:
        """
        获取单个股票的面板数据
        :param df: 完整数据集
        :param stock_id: 股票ID，如果为None则返回所有股票
        :return: 股票面板数据
        """
        if stock_id is not None:
            return df[df['instrument_id'] == stock_id].copy()
        return df


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化特征工程
        :param df: 原始数据
        """
        self.df = df.copy()
        
    def create_basic_features(self) -> pd.DataFrame:
        """
        创建基础特征
        :return: 包含基础特征的DataFrame
        """
        df = self.df.copy()
        
        # 按股票分组计算特征
        for stock_id in df['instrument_id'].unique():
            mask = df['instrument_id'] == stock_id
            stock_data = df[mask].copy()
            
            # 价格相关特征
            df.loc[mask, 'price_range'] = (stock_data['high'] - stock_data['low']) / stock_data['close']
            df.loc[mask, 'price_position'] = (stock_data['close'] - stock_data['low']) / (stock_data['high'] - stock_data['low'] + 1e-8)
            
            # 成交量相关特征
            df.loc[mask, 'volume_ma5'] = stock_data['volume'].rolling(5).mean()
            df.loc[mask, 'volume_ma10'] = stock_data['volume'].rolling(10).mean()
            df.loc[mask, 'volume_ma20'] = stock_data['volume'].rolling(20).mean()
            
            # 价格移动平均
            df.loc[mask, 'close_ma5'] = stock_data['close'].rolling(5).mean()
            df.loc[mask, 'close_ma10'] = stock_data['close'].rolling(10).mean()
            df.loc[mask, 'close_ma20'] = stock_data['close'].rolling(20).mean()
            
            # 波动率
            df.loc[mask, 'volatility_5'] = stock_data['returns'].rolling(5).std()
            df.loc[mask, 'volatility_10'] = stock_data['returns'].rolling(10).std()
            df.loc[mask, 'volatility_20'] = stock_data['returns'].rolling(20).std()
            
            # 动量特征
            df.loc[mask, 'momentum_5'] = stock_data['close'].pct_change(5)
            df.loc[mask, 'momentum_10'] = stock_data['close'].pct_change(10)
            df.loc[mask, 'momentum_20'] = stock_data['close'].pct_change(20)
            
        return df
    
    def create_market_features(self) -> pd.DataFrame:
        """
        创建市场相关特征（横截面特征）
        :return: 包含市场特征的DataFrame
        """
        df = self.df.copy()
        
        # 按日期分组计算市场特征
        for date in df['date'].unique():
            mask = df['date'] == date
            date_data = df[mask].copy()
            
            # 市值排名
            df.loc[mask, 'market_cap_rank'] = date_data['total_capital'].rank(pct=True)
            
            # 成交量排名
            df.loc[mask, 'volume_rank'] = date_data['volume'].rank(pct=True)
            
            # 换手率排名
            df.loc[mask, 'turnover_rate_rank'] = date_data['turnover_rate'].rank(pct=True)
            
        return df


def prepare_data_for_model(df: pd.DataFrame, 
                           feature_cols: list,
                           target_col: str = 'y') -> Tuple[pd.DataFrame, pd.Series]:
    """
    准备用于模型训练的数据
    :param df: 完整数据集
    :param feature_cols: 特征列名列表
    :param target_col: 目标列名
    :return: (特征DataFrame, 目标Series)
    """
    # 移除包含NaN的行
    df_clean = df.dropna(subset=feature_cols + [target_col])
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    return X, y

