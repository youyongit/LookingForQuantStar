# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha31(bars,fast=False):
    """Alpha31 of 101 Alphas(10分钟涨跌的线性加权均值+3分钟涨跌+成交量和最低相关性符号)"""
    close = bars['close']
    low = bars['low']
    volume = bars['volume']
    adv20 = sma(volume, 20)
    df = correlation(adv20, low, 12)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    return decay_linear(-1 * delta(close, 10), 10) + -1 * delta(close, 3) + sign(df)

