# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha24(bars,fast=False): 
    """Alpha24 of 101 Alphas"""
    n = 100
    close = bars['close']
    cond = delta(sma(close, n), n) / delay(close, n) <= 0.05
    alpha = -1 * delta(close, 3)
    alpha[cond] = -1 * (close - ts_min(close, n))
    return alpha

