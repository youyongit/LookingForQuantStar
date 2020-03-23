
'''
从 data 中解压并获得K线

K线还要注意对齐

'''

# 路径管理(先采用相对路径)
import sys,os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np


class sim_redis(object):
    '''模拟的redis'''
    def __init__(self, maxWindow=3000):
        self.kMap = {}
        self.maxWindow = maxWindow

    def lrange(self, key, num):
        if key in self.kMap:
            return self.kMap[key][-num:]
        else:
            return []

    def on_bar(self, bar):
        symbol = bar[7]
        if symbol not in self.kMap:
            self.kMap[symbol] = []
        self.kMap[symbol].append(bar)
        if len(self.kMap[symbol])>self.maxWindow:
            del self.kMap[symbol][0]

    def llen(self, key, num):
        if key in self.kMap:
            return len(self.kMap[key])
        else:
            return 0

if __name__ == '__main__':

    from common.get_data import get_data

    from time import time
    import pandas as pd

    r = sim_redis()

    t1 = time()
    for bar in get_data(symbols=['btc']):
        r.on_bar(bar)
    t2 = time()

    print(t2-t1)

    print(pd.DataFrame(r.lrange('btc', 3000)))