import pandas as pd 
import numpy as np
#from ctaFunction import std_normalized

def barStdNormal(bars, timeperiod=5):
    '''Std Normal '''
    close = bars['close']
    result = close.rolling(timeperiod).apply(std_normalized)

    return result