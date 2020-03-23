# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha14(bars,fast=False):
    """Alpha14 of 101 Alphas"""
    open = bars['open']
    volume = bars['volume']
    returns = bars['close']/bars['close'].shift(1)-1
    df = correlation(open, volume, 10)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    return -1 * (delta(returns, 3)) * df
