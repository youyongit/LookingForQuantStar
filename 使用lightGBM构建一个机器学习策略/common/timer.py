import time
'''
    计算函数运行时间的包装器
'''


def print_run_time(func):  
    def wrapper(*args, **kw):  
        local_time = time.time()  
        func(*args, **kw) 
        print ('current Function [%s] run time is %.2f' % (func.__name__ ,time.time() - local_time))
    return wrapper
