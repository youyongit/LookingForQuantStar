# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha4(bars,fast=False):
    """Alpha4 of 101 Alphas"""
    low = bars['low']
    return -1 * ts_rank(low, 9)
