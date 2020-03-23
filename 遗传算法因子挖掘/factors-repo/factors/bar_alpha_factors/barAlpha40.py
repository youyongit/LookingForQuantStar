# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *
#---------------------------------------------------------------
def barAlpha40(bars,fast=False):
    """Alpha40 of 101 Alphas"""
    high = bars['high']
    volume = bars['volume']
    return -1 * (stddev(high, 10)) * correlation(high, volume, 10)
