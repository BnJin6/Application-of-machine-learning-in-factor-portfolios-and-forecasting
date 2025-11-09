# -*- coding: utf-8 -*-
"""
日志记录模块
"""
import logging
import os
from datetime import datetime


class Logger:
    """日志记录器"""
    
    def __init__(self, name='stock_prediction', log_dir='logs'):
        """
        初始化日志记录器
        :param name: 日志名称
        :param log_dir: 日志目录
        """
        self.name = name
        self.log_dir = log_dir
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名（带时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        # 配置日志
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """配置日志记录器"""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        # 清除已有的处理器
        logger.handlers.clear()
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def debug(self, message):
        """记录DEBUG级别日志"""
        self.logger.debug(message)
    
    def warning(self, message):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def critical(self, message):
        """记录CRITICAL级别日志"""
        self.logger.critical(message)
    
    def log_separator(self, char='=', length=80):
        """记录分隔线"""
        self.info(char * length)
    
    def log_section(self, title, char='=', length=80):
        """记录章节标题"""
        self.log_separator(char, length)
        self.info(title)
        self.log_separator(char, length)
    
    def log_dict(self, data_dict, title=None):
        """记录字典数据"""
        if title:
            self.info(title)
        for key, value in data_dict.items():
            self.info(f"  {key}: {value}")
    
    def log_dataframe_info(self, df, name='DataFrame'):
        """记录DataFrame信息"""
        self.info(f"{name} 信息:")
        self.info(f"  形状: {df.shape}")
        self.info(f"  列数: {len(df.columns)}")
        self.info(f"  行数: {len(df)}")
        self.info(f"  内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    def get_log_file(self):
        """获取日志文件路径"""
        return self.log_file


def create_logger(name='stock_prediction', log_dir='logs'):
    """
    创建日志记录器的便捷函数
    :param name: 日志名称
    :param log_dir: 日志目录
    :return: Logger实例
    """
    return Logger(name, log_dir)

