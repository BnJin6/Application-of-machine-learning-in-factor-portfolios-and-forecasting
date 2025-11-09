# -*- coding: utf-8 -*-
"""
测试数据加载
"""
import pandas as pd

print("开始测试数据加载...")

# 尝试多种编码方式
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'iso-8859-1']

for encoding in encodings:
    try:
        print(f"\n尝试使用 {encoding} 编码...")
        df = pd.read_csv('daily_data.csv', encoding=encoding, nrows=10)
        print(f"成功！读取了 {len(df)} 行")
        print(f"列名: {list(df.columns)}")
        print(f"\n前3行数据:")
        print(df.head(3))
        break
    except Exception as e:
        print(f"失败: {str(e)[:100]}")
        continue

print("\n测试完成")

