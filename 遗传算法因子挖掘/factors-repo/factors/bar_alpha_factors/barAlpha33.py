# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha33(bars,fast=False):
    """Alpha33 of 101 Alphas"""
    close = bars['close']
    open = bars['open']
    return (-1 + (open / close))
