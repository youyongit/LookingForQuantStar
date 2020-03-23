# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha45(bars,fast=False):
    """Alpha45 of 101 Alphas"""
    close = bars['close']
    volume = bars['volume']
    df = correlation(close, volume, 2)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    return -1 * ((sma(delay(close, 5), 20)) * df * (correlation(ts_sum(close, 5), ts_sum(close, 20), 2)))
