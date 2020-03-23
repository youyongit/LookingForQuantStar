import sys,os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)
sys.path.append('/home/jaq/project/simpleback/')
sys.path.append('/home/jaq/project/simpleback/mlutils')

import time
import json
import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

from common.get_data import get_data_df
from mlutils.factorsLoader import *
from mlutils.label import *

from multiprocessing import Pool

from sklearn.preprocessing import Imputer,MinMaxScaler,scale
from sklearn.linear_model import LinearRegression,Ridge,BayesianRidge, Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, f_regression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,BaggingRegressor

import lightgbm as lgb

factors_name = [f.split('/')[-1] for f in ALL_FACTORS]
check_name = ['f_regression', 'linear_reg',  'ridge', 'lasso',
              'svr', 'ref', 'random_forest', 'lgboost']
factors = sys.modules['Factors']
#print(sys.modules)

def cal_factors_from_data(klineDf):

    # 由CPU可用核心数决定
    factors_pool = Pool(5)
    
    result = {}
    for f in factors_name:
        fobj = getattr(factors, f)
        result[f] = factors_pool.apply_async(fobj, args=(klineDf,))

    factors_pool.close()
    factors_pool.join()

    for f in factors_name:
        klineDf[f] = result[f].get()

    return klineDf

def load_kline_data(symbol, period='5min'):
    '''
    从数据库中导入全部数据
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
    #print(df)
    return df


def draw_heatmap(df, symbol, period='5min'):
    dfData = df.corr()
    plt.subplots(figsize=(144, 144)) # 设置画面大小
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
    plt.savefig('./factors_analysis/corr/{}_{}.png'.format(symbol, period))
    #plt.show()

if __name__ == "__main__":

    symbols_list = ['btc', 'eth', 'eos', 'ltc']
    periods = ['1min', '5min', '30min', '60min']

    for s in symbols_list:
        for p in periods:
            kline_df = load_kline_data(s, p)
            kline_df = kline_df[kline_df['volume'] != 0]
            factors_df = cal_factors_from_data(kline_df)
            # ‘pearson’, ‘kendall’, ‘spearman’
            pearson_corr_df = factors_df.corr('pearson')
            pearson_corr_df.to_csv('./factors_analysis/corr/pearson_{}_{}.csv'.format(s, p))

            kendall_corr_df = factors_df.corr('kendall')
            kendall_corr_df.to_csv('./factors_analysis/corr/kendall_{}_{}.csv'.format(s, p))

            spearman_corr_df = factors_df.corr('kendall')
            spearman_corr_df.to_csv('./factors_analysis/corr/kendall_{}_{}.csv'.format(s, p))

            draw_heatmap(factors_df, s, period=p)