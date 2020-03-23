
'''
从 data 中解压并获得K线

K线还要注意对齐

'''

# 路径管理(先采用相对路径)
import sys,os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

import pandas as pd
import numpy as np
import gzip

def load_kline_data(symbol, period='5min'):
    '''
    从数据库中导入全部数据 并且提供不同的k线周期
    '''
    df = get_data_df(symbol=symbol)
    df.datetime = pd.to_datetime(df['datetime']*1e9)

    df = df.set_index('datetime')
    df = (df.resample(period)
            .agg({'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'}))
    df = df.reset_index(drop=False)
    
    return df

def zip_csv(dbname='okex_q'):
    fList = os.listdir(os.path.join(root_path, 'data', dbname))
    for fname in fList:
        if '.csv' not in fname:
            continue
        zname = f'{fname}.gzip'
        zpath = os.path.join(root_path, 'data', dbname, zname)
        fpath = os.path.join(root_path, 'data', dbname, fname)
        with open(fpath, 'rb') as f_in:
            with gzip.open(zpath, 'wb') as f_out:
                f_out.write(f_in.read())

def get_data_df(dbname='okex_q', symbol='btc'):
    fList = os.listdir(os.path.join(root_path, 'data', dbname))
    symbol_map = {}
    for fname in fList:
        symbol_tmp = fname.split('_')[0]
        symbol_map[symbol_tmp.lower()] = os.path.join(root_path, 'data', dbname, fname)

    fname = symbol_map[symbol.lower()]
    df = pd.read_csv(fname, compression='gzip', usecols=[1,2,3,4,5,6])
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    return df

def get_data(dbname='okex_q', symbols=['btc', 'eos', 'etc', 'eth', 'ltc']):
    ''''''
    fList = os.listdir(os.path.join(root_path, 'data', dbname))
    symbol_map = {}
    for fname in fList:
        symbol = fname.split('_')[0]
        symbol_map[symbol.lower()] = os.path.join(root_path, 'data', dbname, fname)

    next_bar    = [[]    for tmp in symbols]
    current_bar = [[]    for tmp in symbols]

    finish_list = [False for tmp in symbols]
    iter_list   = [None  for tmp in symbols]   # 迭代器的初始化

    data_map = {}
    for si, symbol in enumerate(symbols):
        fname = symbol_map[symbol.lower()]
        df = pd.read_csv(fname, compression='gzip', usecols=[1,2,3,4,5,6])
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        data_map[symbol] = df
        iter_list[si] = iter_one(df, symbol)
    
    while True:

        finished_flag = np.all(finish_list)
        if finished_flag:
            break
        
        for index, value in enumerate(iter_list):

            if finish_list[index]:
                continue

            symbol = symbols[index]

            next_dt = [tmp[0] for tmpI, tmp in enumerate(next_bar) if tmp and not finish_list[tmpI]]
            if next_dt:
                next_dt_min = np.min(next_dt)
                if next_bar[index] and next_bar[index][0]>next_dt_min:
                    # 此时不是最快到来的未来，可以继续等待
                    continue
            
            try:
                bar = next(value)
                next_bar[index] = bar
            except StopIteration:
                finish_list[index] = True
                continue

            current_bar[index] = next_bar[index]
            yield current_bar[index]



def iter_one(df, symbol):
    for row in df.itertuples():
        if row.volume>0:
            bar = [row.datetime, row.open, row.high, row.low, row.close, row.volume, row.volume/row.close, symbol]
            yield bar

if __name__ == "__main__":

    from time import time

    t1 = time()
    for bar in get_data(symbols=['btc', 'eos', 'eth', 'etc', 'ltc']):
        pass
    t2 = time()

    print(t2-t1)

    print(bar)
