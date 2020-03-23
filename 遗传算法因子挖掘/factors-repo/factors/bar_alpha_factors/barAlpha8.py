# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha8(bars,fast=False):
    """Alpha8 of 101 Alphas"""
    open = bars['open']
    returns = bars['close']/bars['close'].shift(1)-1
    return -1 * ((ts_sum(open, 5) * ts_sum(returns, 5)) -\
                    delay((ts_sum(open, 5) * ts_sum(returns, 5)), 10))
