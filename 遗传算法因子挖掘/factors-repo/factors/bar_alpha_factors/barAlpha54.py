# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha54(bars,fast=False):
    """Alpha54 of 101 Alphas"""
    close = bars['close']
    open = bars['open']
    high = bars['high']
    low = bars['low']
    inner = (low - high).replace(0, -0.0001)
    return -1 * (low - close) / inner * (open ** 5/close ** 5)

