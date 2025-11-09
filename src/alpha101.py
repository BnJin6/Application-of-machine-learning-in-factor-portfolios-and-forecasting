# -*- coding: utf-8 -*-
"""
Alpha101因子实现
基于WorldQuant的101个Alpha因子
"""
import numpy as np
import pandas as pd
from src.utils import *


class Alpha101:
    """
    Alpha101因子计算类
    实现WorldQuant的101个量化因子
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化Alpha101
        :param data: 包含OHLCV数据的DataFrame
        """
        self.data = data.copy()
        self.prepare_basic_data()
        
    def prepare_basic_data(self):
        """准备基础数据"""
        # 按股票分组
        self.grouped = self.data.groupby('instrument_id')
        
        # 为每个股票创建基础变量
        self.close = self.data.pivot(index='date', columns='instrument_id', values='close')
        self.open = self.data.pivot(index='date', columns='instrument_id', values='open')
        self.high = self.data.pivot(index='date', columns='instrument_id', values='high')
        self.low = self.data.pivot(index='date', columns='instrument_id', values='low')
        self.volume = self.data.pivot(index='date', columns='instrument_id', values='volume')
        self.vwap = self.data.pivot(index='date', columns='instrument_id', values='vwap')
        self.returns = self.data.pivot(index='date', columns='instrument_id', values='returns')
        
        # 计算常用的平均成交量
        self.adv5 = ts_mean(self.volume, 5)
        self.adv10 = ts_mean(self.volume, 10)
        self.adv15 = ts_mean(self.volume, 15)
        self.adv20 = ts_mean(self.volume, 20)
        self.adv30 = ts_mean(self.volume, 30)
        self.adv40 = ts_mean(self.volume, 40)
        self.adv50 = ts_mean(self.volume, 50)
        self.adv60 = ts_mean(self.volume, 60)
        self.adv81 = ts_mean(self.volume, 81)
        self.adv120 = ts_mean(self.volume, 120)
        self.adv150 = ts_mean(self.volume, 150)
        self.adv180 = ts_mean(self.volume, 180)
    
    # ==================== Alpha因子实现 ====================
    
    def alpha001(self):
        """
        Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        """
        condition = self.returns < 0
        part1 = condition * ts_std(self.returns, 20) + (~condition) * self.close
        alpha = rank(ts_argmax(signed_power(part1, 2), 5)) - 0.5
        return handle_inf_nan(alpha)
    
    def alpha002(self):
        """
        Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        """
        df = -1 * correlation(rank(delta(np.log(self.volume), 2)), 
                             rank((self.close - self.open) / self.open), 6)
        return handle_inf_nan(df)
    
    def alpha003(self):
        """
        Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
        """
        alpha = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return handle_inf_nan(alpha)
    
    def alpha004(self):
        """
        Alpha#4: (-1 * Ts_Rank(rank(low), 9))
        """
        alpha = -1 * ts_rank(rank(self.low), 9)
        return handle_inf_nan(alpha)
    
    def alpha005(self):
        """
        Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
        """
        alpha = rank(self.open - ts_mean(self.vwap, 10)) * (-1 * np.abs(rank(self.close - self.vwap)))
        return handle_inf_nan(alpha)
    
    def alpha006(self):
        """
        Alpha#6: (-1 * correlation(open, volume, 10))
        """
        alpha = -1 * correlation(self.open, self.volume, 10)
        return handle_inf_nan(alpha)
    
    def alpha007(self):
        """
        Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
        """
        condition = self.adv20 < self.volume
        part1 = -1 * ts_rank(np.abs(delta(self.close, 7)), 60) * np.sign(delta(self.close, 7))
        alpha = condition * part1 + (~condition) * (-1)
        return handle_inf_nan(alpha)
    
    def alpha008(self):
        """
        Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
        """
        sum_open = ts_sum(self.open, 5)
        sum_returns = ts_sum(self.returns, 5)
        alpha = -1 * rank(sum_open * sum_returns - delay(sum_open * sum_returns, 10))
        return handle_inf_nan(alpha)
    
    def alpha009(self):
        """
        Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : 
                  ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
        """
        delta_close = delta(self.close, 1)
        condition1 = ts_min(delta_close, 5) > 0
        condition2 = ts_max(delta_close, 5) < 0
        alpha = condition1 * delta_close + condition2 * delta_close + (~condition1 & ~condition2) * (-1 * delta_close)
        return handle_inf_nan(alpha)
    
    def alpha010(self):
        """
        Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : 
                       ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
        """
        delta_close = delta(self.close, 1)
        condition1 = ts_min(delta_close, 4) > 0
        condition2 = ts_max(delta_close, 4) < 0
        inner = condition1 * delta_close + condition2 * delta_close + (~condition1 & ~condition2) * (-1 * delta_close)
        alpha = rank(inner)
        return handle_inf_nan(alpha)
    
    def alpha011(self):
        """
        Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
        """
        alpha = (rank(ts_max(self.vwap - self.close, 3)) + 
                rank(ts_min(self.vwap - self.close, 3))) * rank(delta(self.volume, 3))
        return handle_inf_nan(alpha)
    
    def alpha012(self):
        """
        Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        """
        alpha = np.sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))
        return handle_inf_nan(alpha)
    
    def alpha013(self):
        """
        Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))
        """
        alpha = -1 * rank(covariance(rank(self.close), rank(self.volume), 5))
        return handle_inf_nan(alpha)
    
    def alpha014(self):
        """
        Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        """
        alpha = -1 * rank(delta(self.returns, 3)) * correlation(self.open, self.volume, 10)
        return handle_inf_nan(alpha)
    
    def alpha015(self):
        """
        Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
        """
        alpha = -1 * ts_sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3)
        return handle_inf_nan(alpha)
    
    def alpha016(self):
        """
        Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))
        """
        alpha = -1 * rank(covariance(rank(self.high), rank(self.volume), 5))
        return handle_inf_nan(alpha)
    
    def alpha017(self):
        """
        Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * 
                   rank(ts_rank((volume / adv20), 5)))
        """
        alpha = ((-1 * rank(ts_rank(self.close, 10))) * 
                rank(delta(delta(self.close, 1), 1)) * 
                rank(ts_rank(self.volume / self.adv20, 5)))
        return handle_inf_nan(alpha)
    
    def alpha018(self):
        """
        Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
        """
        alpha = -1 * rank(ts_std(np.abs(self.close - self.open), 5) + 
                         (self.close - self.open) + 
                         correlation(self.close, self.open, 10))
        return handle_inf_nan(alpha)
    
    def alpha019(self):
        """
        Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * 
                   (1 + rank((1 + sum(returns, 250)))))
        """
        alpha = (-1 * np.sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) * \
                (1 + rank(1 + ts_sum(self.returns, 250)))
        return handle_inf_nan(alpha)
    
    def alpha020(self):
        """
        Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * 
                   rank((open - delay(low, 1))))
        """
        alpha = ((-1 * rank(self.open - delay(self.high, 1))) * 
                rank(self.open - delay(self.close, 1)) * 
                rank(self.open - delay(self.low, 1)))
        return handle_inf_nan(alpha)
    
    def alpha021(self):
        """
        Alpha#21: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : 
                   (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : 
                   (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))
        """
        ma8 = ts_mean(self.close, 8)
        std8 = ts_std(self.close, 8)
        ma2 = ts_mean(self.close, 2)
        vol_ratio = self.volume / self.adv20
        
        condition1 = (ma8 + std8) < ma2
        condition2 = ma2 < (ma8 - std8)
        condition3 = vol_ratio >= 1
        
        alpha = condition1 * (-1) + (~condition1 & condition2) * 1 + \
                (~condition1 & ~condition2 & condition3) * 1 + \
                (~condition1 & ~condition2 & ~condition3) * (-1)
        return handle_inf_nan(alpha)
    
    def alpha022(self):
        """
        Alpha#22: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
        """
        alpha = -1 * delta(correlation(self.high, self.volume, 5), 5) * rank(ts_std(self.close, 20))
        return handle_inf_nan(alpha)
    
    def alpha023(self):
        """
        Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
        """
        condition = ts_mean(self.high, 20) < self.high
        alpha = condition * (-1 * delta(self.high, 2))
        return handle_inf_nan(alpha)
    
    def alpha024(self):
        """
        Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || 
                   ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? 
                   (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
        """
        ma100 = ts_mean(self.close, 100)
        condition = (delta(ma100, 100) / delay(self.close, 100)) <= 0.05
        alpha = condition * (-1 * (self.close - ts_min(self.close, 100))) + \
                (~condition) * (-1 * delta(self.close, 3))
        return handle_inf_nan(alpha)
    
    def alpha025(self):
        """
        Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
        """
        alpha = rank((-1 * self.returns) * self.adv20 * self.vwap * (self.high - self.close))
        return handle_inf_nan(alpha)
    
    def alpha026(self):
        """
        Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        """
        alpha = -1 * ts_max(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3)
        return handle_inf_nan(alpha)
    
    def alpha101(self):
        """
        Alpha#101: ((close - open) / ((high - low) + 0.001))
        简单但有效的价格位置因子
        """
        alpha = (self.close - self.open) / (self.high - self.low + 0.001)
        return handle_inf_nan(alpha)
    
    # ==================== 批量计算函数 ====================
    
    def calculate_all_alphas(self, alpha_list=None):
        """
        计算所有Alpha因子
        :param alpha_list: 要计算的alpha列表，如果为None则计算所有
        :return: 包含所有alpha因子的字典
        """
        if alpha_list is None:
            # 默认计算前26个alpha和alpha101
            alpha_list = list(range(1, 27)) + [101]
        
        results = {}
        
        for alpha_num in alpha_list:
            method_name = f'alpha{alpha_num:03d}'
            if hasattr(self, method_name):
                print(f"正在计算 {method_name}...")
                try:
                    results[method_name] = getattr(self, method_name)()
                except Exception as e:
                    print(f"计算 {method_name} 时出错: {str(e)}")
                    results[method_name] = None
        
        return results
    
    def get_alpha_dataframe(self, alpha_dict):
        """
        将alpha字典转换为DataFrame格式
        :param alpha_dict: alpha因子字典
        :return: 长格式的DataFrame
        """
        result_list = []
        
        for alpha_name, alpha_values in alpha_dict.items():
            if alpha_values is not None:
                # 将宽格式转换为长格式
                df_long = alpha_values.stack().reset_index()
                df_long.columns = ['date', 'instrument_id', alpha_name]
                result_list.append(df_long)
        
        # 合并所有alpha因子
        if result_list:
            result_df = result_list[0]
            for df in result_list[1:]:
                result_df = result_df.merge(df, on=['date', 'instrument_id'], how='outer')
            return result_df
        
        return pd.DataFrame()

