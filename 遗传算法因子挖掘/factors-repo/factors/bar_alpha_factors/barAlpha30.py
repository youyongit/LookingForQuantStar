# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha30(bars,fast=False):
    """Alpha30 of 101 Alphas"""
    close = bars['close']
    volume = bars['volume']
    delta_close = delta(close, 1)
    inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
    return ((1.0 - (inner)) * ts_sum(volume, 5)) / ts_sum(volume, 20)
