# 导入需要的包工具

import numpy as np
import pandas as pd
import graphviz
from scipy.stats import rankdata
import pickle
from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

# 从Co
from common.get_data import load_kline_data

from utils.fitness import * 
from utils.functions import *
from utils.utils import *

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    df_data = load_kline_data('btc', period='5min')
    df_data = df_data[df_data['volume'] != 0]
    
    fields = ['open', 'high', 'low', 'close', 'volume']

    X = df_data[fields].fillna(0).replace(np.inf,0).replace(-np.inf,0)
    y = (df_data['close'].shift(-1) / df_data['close']) -1
    y = y.fillna(0).replace(np.inf,0).replace(-np.inf,0)

    train_X, test_X, train_y, test_y = train_test_split(X, y)

    generations = 50
    function_set = init_function + user_function
    metric = MSLE
    population_size = 100
    random_state=0
    est_gp = SymbolicTransformer(
                                feature_names=fields, 
                                function_set=function_set,
                                generations=generations,
                                metric=metric,
                                population_size=population_size,
                                tournament_size=20, 
                                random_state=random_state,
                            )

    est_gp.fit(train_X, train_y)

    best_programs = est_gp._best_programs
    best_programs_dict = {}
    for p in best_programs:
        factor_name =  str(best_programs.index(p) + 1)
        best_programs_dict[factor_name] = {'fitness':p.fitness_, 'expression':str(p), 'depth':p.depth_, 'length':p.length_}
        
    best_programs_dict = pd.DataFrame(best_programs_dict).T
    best_programs_dict = best_programs_dict.sort_values(by='fitness')
    
    