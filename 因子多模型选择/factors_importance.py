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

from common.get_data import get_data_df
from mlutils.factorsLoader import *
from mlutils.label import *

from sklearn.preprocessing import Imputer,MinMaxScaler,scale
from sklearn.linear_model import LinearRegression,Ridge,BayesianRidge, Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, f_regression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,BaggingRegressor
from sklearn.model_selection import train_test_split
import lightgbm as lgb

factors_name = [f.split('/')[-1] for f in ALL_FACTORS]
check_name = ['linear_reg',  'ridge', 'lasso',
              'ref', 'random_forest', 'lgboost']
factors = sys.modules['Factors']
#print(sys.modules)

from multiprocessing import Pool

class MLCHECK:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranks = {}

    def f_regression_check(self, X, Y, names):
        print ('calc f_regression importance {}'.format(os.getpid()))
        f, pval  = f_regression(X, Y, center=True)
        f[np.isnan(f)] = 0
        #self.ranks["Corr."] =
        print ('calc f_regression finished !')
        return rank_to_dict(f, names)

    def linear_reg_check(self, X, Y, names):
        print ('calc Linear reg importance {}'.format(os.getpid()))
        lr = LinearRegression(normalize=True)
        lr.fit(X, Y)
        #self.ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)
        print ('calc Linear reg importance finished !')
        return rank_to_dict(np.abs(lr.coef_), names)

    def ridge_check(self, X, Y, names):
        print ('calc Ridge importance {}'.format(os.getpid()))
        ridge = Ridge(alpha=10)
        ridge.fit(X, Y)
        #self.ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)
        print ('calc ridge importance finished !')
        return rank_to_dict(np.abs(ridge.coef_), names)

    def lasso_check(self, X, Y, names):
        print ('calc Lasso importance {}'.format(os.getpid()))
        lasso = LassoCV(alphas = np.logspace(0,-9,10))
        lasso.fit(X, Y)
        #self.ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)
        print ('calc Lasso importance finished !')
        return rank_to_dict(np.abs(lasso.coef_), names)

    def svr_check(self, X, Y, names):
        print('calc SVR importance {}'.format(os.getpid()))
        svr = SVR(kernel="linear")
        svr.fit(X, Y) 
        #self.rank["SVR"] = rank_to_dict(np.abs(svr.coef_), names)
        print ('calc SVR importance finished !')
        return rank_to_dict(np.abs(svr.coef_), names)

    def ref_check(self, X, Y, names):
        print ('calc RFE importance {}'.format(os.getpid()))
        #stop the search when 5 features are left (they will get equal scores)
        ridge = Ridge(alpha=10)
        rfe = RFE(ridge, n_features_to_select=5)
        rfe.fit(X,Y)
        #self.ranks["RFE"] = rank_to_dict(rfe.ranking_, names, order=-1)
        print ('calc RFE importance finished !')
        return rank_to_dict(rfe.ranking_, names, order=-1)

    def random_forest_check(self, X, Y, names):
        print ('calc RandomForest importance {}'.format(os.getpid()))
        rf = RandomForestRegressor()
        rf.fit(X,Y)
        #self.ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
        print ('calc random forest importance finished !')
        return rank_to_dict(rf.feature_importances_, names)

    def lgboost_check(self, X, Y, names):
        print('calc lgb importance {}'.format(os.getpid()))
        lgboost = lgb.LGBMRegressor(num_leaves=10,
                            learning_rate=0.05,
                            n_estimators=200)
        lgboost.fit(X, Y)
        #self.ranks["LGB"] = rank_to_dict(lgboost.feature_importances_, names)     
        print ('calc lgb importance finished !')
        return  rank_to_dict(lgboost.feature_importances_, names)     

def cal_factors_from_data(klineDf):

    # 由CPU可用核心数决定
    factors_pool = Pool(10)
    
    result = {}
    for f in factors_name:
        fobj = getattr(factors, f)
        result[f] = factors_pool.apply_async(fobj, args=(klineDf,))

    factors_pool.close()
    factors_pool.join()

    for f in factors_name:
        klineDf[f] = result[f].get()

    return klineDf

def mp_check_func(ml_check, X, Y, names):
    rank = pd.DataFrame()
    result = {}
    p = Pool(10)
    for c in check_name:
        fobj = getattr(ml_check, c + '_check')
        result[c] = p.apply_async(fobj, args=(X, Y, names))

    p.close()
    p.join()

    for c in check_name:
        rank[c] = result[c].get()

    return rank

def cal_label_from_data(klineDf):
    
    labels = fixed_time_horizon_ma(klineDf)
    #self.labels = volatilitty_time(self.klineDf)
    return labels

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 4), ranks)
    return pd.Series(dict(zip(names, ranks )))

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

def importance_check(X, Y, names):
    ml_check = MLCHECK()

    ranks_df = mp_check_func(ml_check, X, Y, names)
    
    dfRank = pd.DataFrame(ranks_df)
    dfRank['Mean'] = dfRank.mean(axis=1)

    return dfRank

    

def check_one_symbol(symbol='btc', period='5min'):
    data_df = load_kline_data(symbol='eth', period='5min')
    data_df = data_df[data_df['volume'] != 0]

    factors = cal_factors_from_data(data_df).drop(['datetime'], axis=1)
    factors['label'] = cal_label_from_data(data_df)

    factors = factors.fillna(0).replace(np.inf,0).replace(-np.inf,0)

    labels = factors['label']
    factors = factors[factors_name]

    print(factors.columns.values.tolist())
    # 执行检查
    dfRank = importance_check(factors, labels, factors_name)

    dfRank.to_csv('./factors_analysis/importance/model_importance_{}_{}.csv'.format(symbol, period))


if __name__ == "__main__":

    symbols_list = ['btc', 'eth', 'eos', 'ltc']
    periods = ['1min', '5min']

    for symbol in symbols_list:
        for period in periods:
            check_one_symbol(symbol, period)