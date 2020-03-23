# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import talib
import numpy as np
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha61(bars,fast=False):
    """根据基因工程生成 （PH5*PH5*FRV80*FRV80）"""
    n = 80
    n0 = 5
    high = bars['high'].rolling(n0).max()
    close = bars['close']
    volume  = bars['volume'].diff().values.astype(np.float64)
    d = (high-close)/close
    x = talib.CORREL(close.values,volume,n)
    return  d * d * x * x
