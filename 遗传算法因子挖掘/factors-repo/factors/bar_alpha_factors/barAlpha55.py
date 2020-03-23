# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha55(bars,fast=False):
    """Alpha55 of 101 Alphas"""
    close = bars['close']
    high = bars['high']
    low = bars['low']
    volume = bars['volume']
    divisor = (ts_max(high, 12) - ts_min(low, 12)).replace(0, 0.0001)
    inner = (close - ts_min(low, 12)) / (divisor)
    df = correlation(inner, volume, 6)
    return -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)

