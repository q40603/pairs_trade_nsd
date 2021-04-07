import json
import numpy as np
import pandas as pd
import requests
import datetime
import sys

def sb_pred(date='20200904',stock1='1402',stock2='2912'):
    if date =="":
        date = datetime.date.today()
        while (date.weekday()>0):
            date -= datetime.timedelta(days=1)
        date = date.strftime("%Y%m%d")
    func_para = '{"date":'+date+',"stock1":'+stock1+',"stock2":'+stock2+'}'
    
    res = requests.post('http://mpcdl.cs.nctu.edu.tw:5019/predict/api/', json=func_para)
    print(res)
    return res.json()


a = sb_pred(date=sys.argv[1],stock1=sys.argv[2],stock2=sys.argv[3])  #將日期跟stock id(小的在前面，大的在後面)傳入這個func 會回傳當天115分鐘(150~265)的發生結構性改變的機率，如果沒有找到這個pair，會回傳全部都是0的list
print(a)
