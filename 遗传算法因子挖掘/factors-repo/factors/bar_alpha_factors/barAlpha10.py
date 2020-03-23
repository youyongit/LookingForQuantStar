# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha10(bars, fast=False):
    """Alpha10 of 101 Alphas"""
    close = bars['close']
    delta_close = delta(close, 1)
    cond_1 = ts_min(delta_close, 4) > 0
    cond_2 = ts_max(delta_close, 4) < 0
    alpha = -1 * delta_close
    alpha[cond_1 | cond_2] = delta_close
    return alpha
