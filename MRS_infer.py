# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch
import sys
import torch.nn as nn

import pymysql
import os
import json

#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine

from sklearn import preprocessing
import torch.utils.data as Data

from ResNet.model import MSResNet

db_host = '140.113.24.2'
db_name = 'fintech'
db_user = 'fintech'
db_passwd = 'financefintech'
sqlEngine = create_engine('mysql+pymysql://'+db_user+':'+db_passwd+'@'+db_host+'/'+db_name, pool_recycle=3600)
dbConnection = sqlEngine.connect()


fin_db = pymysql.connect(
	host = db_host,
	user = db_user,
	password = db_passwd,
	db = db_name,

)
fin_cursor = fin_db.cursor(pymysql.cursors.DictCursor)



if __name__ == '__main__':
    choose_date= sys.argv[1]
    query = "select left(stime, 19) as mtimestamp, code , sum(volume * price)/(100*sum(volume)) as avg_price from s_price_tick where stime >= '"+ choose_date +" 09:00' and stime <= '"+ choose_date +" 13:30' GROUP BY code, UNIX_TIMESTAMP(mtimestamp)DIV 30;"
    fin_cursor.execute(query)
    result = fin_cursor.fetchall()
    fin_db.commit()
    df = pd.DataFrame(list(result))
    df = df.pivot(index='mtimestamp', columns='code', values='avg_price')
    print(df)
    df = df.fillna(method='ffill')
    df = df.fillna(method='backfill')
    df.index = pd.to_datetime(df.index)
    halfmin = df.resample('30S').mean().copy()

    halfmin = halfmin.reset_index()
    halfmin.index = np.arange(0,len(halfmin),1)
    print(halfmin)

    query = "select stock1, stock2, w1, w2, model_type, snr, zcr, mu, stdev, e_mu, e_stdev from pairs where f_date = '" + choose_date + "';"
    fin_cursor.execute(query)
    table = fin_cursor.fetchall()  

    table = pd.DataFrame(list(table))

    print(table)  


    model = MSResNet(input_channel=3, layers=[1, 1, 1, 1], num_classes=25)
    model.load_state_dict(torch.load("./ResNet/MRS_infer.pkl", map_location=torch.device('cpu')))
    model.eval()
    # model.to(torch.device('cuda:0'))

    whole_day = []
    for pair in range(len(table)):
        spread = table.w1[pair] * np.log((halfmin[ str(table.stock1[pair]) ])) + table.w2[pair] * np.log(halfmin[ str(table.stock2[pair]) ])
        # print(spread)
        spread = spread[32:332].values
        spread = preprocessing.scale(spread)
        new_spread = np.zeros((3,512))
        new_spread[0,106:406] = spread
        
        mindata1 = halfmin[str(table.stock1[pair])][32:332].values
        mindata2 = halfmin[str(table.stock2[pair])][32:332].values

        # print(mindata1)
        # print(mindata2)
        mindata1 = preprocessing.scale(mindata1)
        mindata2 = preprocessing.scale(mindata2)
        
                
        new_spread[1,106:406] = mindata1
        new_spread[2,106:406] = mindata2

        whole_day.append(new_spread)

    whole_day = np.asarray(whole_day)

    whole_day = torch.FloatTensor(whole_day)
    output = model(whole_day)

    if table.shape[0] == 1:
        _, predicted = torch.max(output,0)
    else:
        _, predicted = torch.max(output,1)
    action_choose = predicted.cpu().numpy()
    action_choose = action_choose.tolist()
    table["action"] = pd.Series(action_choose)
    print(table)



    