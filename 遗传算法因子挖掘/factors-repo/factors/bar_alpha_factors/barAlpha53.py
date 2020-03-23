# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha53(bars,fast=False):
    """Alpha53 of 101 Alphas"""
    close = bars['close']
    open = bars['open']
    high = bars['high']
    low = bars['low']
    inner = (close - low).replace(0, 0.0001)
    return -1 * delta((((close - low) - (high - close)) / inner), 9)
