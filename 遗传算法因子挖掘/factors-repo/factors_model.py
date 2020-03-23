import pandas as pd 
import numpy as np 

from common.factors_path import *
from multiprocessing import Pool
import json

factors_name = [f.split('/') for f in ALL_FACTORS]
factors_model = sys.modules['Factors']

class Factors:
    '''
    A mulitProcess Factor calculator

    [init]:
    data_type -- data type, 
        -- 'kline' time candle data 
        -- 'tick'  tick depth 5 data 
        -- 'trade' market order trade data

    gen_type -- generate type.
        gen type is the generate type of factors generator,
        it support two way to generate factors by import data.
        -- 'Increment' support update factors from data generator
        -- 'Full' support update factors from fully dataset

    '''
    def __init__(self, data_type='kline', 
                       gen_type='Full',
                       n_jobs=4):
        super().__init__()
        
        self.data_type = data_type
        self.gen_type = gen_type
        self.factors_info = dict()

        self.n_jobs = n_jobs
    
    def add_factors(self, factors_info):
        self.factors_info = factors_info

    def add_factor(self, factor_name, **kwargs):
        self.factors_info[factor_name] = kwargs

    # TODO Kline 因子计算
    def cal_candle_factors(self, klineDf):
        factors_pool = Pool(processes=self.n_jobs)
        factors_name = self.factors_info.items()

        result = {}
        for factor in factors_name:
            factor_name = factor[0].split('_')[0]
            fobj = getattr(factors_model, factor_name)
            result['{}'.format(factor[0])] = factors_pool.apply_async(fobj, 
                                                                   args=(klineDf,),
                                                                   kwds=factor[1])

        factors_pool.close()
        factors_pool.join()

        for factor in factors_name:
            klineDf['{}'.format(factor[0])] = result['{}'.format(factor[0])].get()

        return klineDf

    # TODO Tick 因子计算
    def cal_tick_factors(self, tickDf):
        pass 

    # # load data interface
    # def load_data(self, data, data_info):
    #     if self.data_type is 'kline':
    #         self.load_candle_data(data, *data_info)
    #     elif self.data_type is 'tick':
    #         self.load_tick_data(data, *data_info)

    # # TODO Kline Data Load 
    # def load_candle_data(self, candleData):
    #     pass 

    # # TODO Tick Data Load
    # def load_tick_data(self):
    #     pass 

if __name__ == "__main__":
    factors_info = {
        'barATR':{'windows': 10},
        'barMaRatio':{'timeperiod': 10}
    }

    factors = Factors(factors_info)
