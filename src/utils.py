# -*- coding: utf-8 -*-
"""
工具函数库 - 用于Alpha因子计算的基础函数
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ==================== 时间序列函数 ====================

def ts_sum(df, window=10):
    """
    滑动窗口数据求和
    :param df: pandas DataFrame
    :param window: 滑动窗口大小
    :return: 滚动求和结果
    """
    return df.rolling(window).sum()


def ts_mean(df, window=10):
    """
    滑动窗口求简单平均数
    :param df: pandas DataFrame
    :param window: 滑动窗口大小
    :return: 滚动平均结果
    """
    return df.rolling(window).mean()


def ts_std(df, window=10):
    """
    滑动窗口求标准差
    :param df: pandas DataFrame
    :param window: 滑动窗口大小
    :return: 滚动标准差结果
    """
    return df.rolling(window).std()


def ts_min(df, window=10):
    """
    滑动窗口中的数据最小值
    :param df: pandas DataFrame
    :param window: 滑动窗口大小
    :return: 滚动最小值结果
    """
    return df.rolling(window).min()


def ts_max(df, window=10):
    """
    滑动窗口中的数据最大值
    :param df: pandas DataFrame
    :param window: 滑动窗口大小
    :return: 滚动最大值结果
    """
    return df.rolling(window).max()


def ts_argmax(df, window=10):
    """
    滑动窗口中的数据最大值位置
    :param df: pandas DataFrame
    :param window: 滑动窗口大小
    :return: 最大值位置
    """
    return df.rolling(window).apply(np.argmax, raw=True) + 1


def ts_argmin(df, window=10):
    """
    滑动窗口中的数据最小值位置
    :param df: pandas DataFrame
    :param window: 滑动窗口大小
    :return: 最小值位置
    """
    return df.rolling(window).apply(np.argmin, raw=True) + 1


def ts_rank(df, window=10):
    """
    滑动窗口中的排序
    :param df: pandas DataFrame
    :param window: 滑动窗口大小
    :return: 时间序列排名
    """
    return df.rolling(window).apply(lambda x: rankdata(x)[-1], raw=True)


# ==================== 数据变换函数 ====================

def delta(df, period=1):
    """
    按参数求一列时间序列数据差值
    :param df: pandas DataFrame
    :param period: 差分周期
    :return: 差分结果
    """
    return df.diff(period)


def delay(df, period=1):
    """
    时间序列数据中第N天前的值
    :param df: pandas DataFrame
    :param period: 延迟周期
    :return: 延迟后的数据
    """
    return df.shift(period)


def rank(df):
    """
    横截面排序，返回排序百分比数
    :param df: pandas DataFrame
    :return: 排名百分比
    """
    return df.rank(axis=1, pct=True)


def scale(df, k=1):
    """
    使df列数据标准化，x绝对值和为k
    :param df: pandas DataFrame
    :param k: 缩放因子
    :return: 标准化后的数据
    """
    return df.mul(k).div(np.abs(df).sum())


# ==================== 统计函数 ====================

def correlation(x, y, window=10):
    """
    滑动窗口求相关系数
    :param x: pandas DataFrame
    :param y: pandas DataFrame
    :param window: 滑动窗口大小
    :return: 相关系数
    """
    return x.rolling(window).corr(y)


def covariance(x, y, window=10):
    """
    滑动窗口求协方差
    :param x: pandas DataFrame
    :param y: pandas DataFrame
    :param window: 滑动窗口大小
    :return: 协方差
    """
    return x.rolling(window).cov(y)


def product(df, window=10):
    """
    滑动窗口中的数据乘积
    :param df: pandas DataFrame
    :param window: 滑动窗口大小
    :return: 滚动乘积
    """
    return df.rolling(window).apply(np.prod, raw=True)


def decay_linear(df, period=10):
    """
    线性加权移动平均
    df中从远及近分别乘以权重1, 2, 3, ..., period，权重和归一化为1
    :param df: pandas DataFrame
    :param period: 加权周期
    :return: 线性衰减加权平均
    """
    if df.isnull().values.any():
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    weights = np.arange(1, period + 1)
    weights = weights / weights.sum()
    
    def weighted_avg(x):
        if len(x) < period:
            return np.nan
        return np.dot(x, weights)
    
    return df.rolling(period).apply(weighted_avg, raw=True)


def signed_power(df, power):
    """
    带符号的幂运算
    :param df: pandas DataFrame
    :param power: 幂次
    :return: 带符号的幂运算结果
    """
    return np.sign(df) * (np.abs(df) ** power)


# ==================== 辅助函数 ====================

def handle_inf_nan(df):
    """
    处理无穷值和NaN值
    :param df: pandas DataFrame
    :return: 处理后的DataFrame
    """
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

