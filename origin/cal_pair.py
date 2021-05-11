import pymysql
import os
import json

import accelerate_formation
import accelerate_trading
import ADF
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from datetime import datetime
from sqlalchemy import create_engine
from vecm_new import para_vecm
from Matrix_function import order_select


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

def spread_mean(stock1,stock2,i,table):
    if table.model_type.iloc[i] == 'model1':
        model = 'H2'
    elif table.model_type.iloc[i] == 'model2':
        model = 'H1*'
    elif table.model_type.iloc[i] == 'model3':
        model = 'H1'
    stock1 = stock1[i,:150]
    stock2 = stock2[i,:150]
    b1 = table.w1.iloc[i]
    b2 = table.w2.iloc[i]
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)
    # print(logy)
    lyc = logy.copy()
    p = order_select(logy,5)
    #print('p:',p)
    _,_,para = para_vecm(logy,model,p)
    logy = np.mat(logy)
    y_1 = np.mat(logy[p:])
    dy = np.mat(np.diff(logy,axis=0))
    for j in range(len(stock1)-p-1):
        if model == 'H1':
            if p!=1:
                delta = para[0] * para[1].T * y_1[j].T + para[2] * np.hstack([dy[j:(j+p-1)].flatten(),np.mat([1])]).T
            else:
                delta = para[0] * para[1].T * y_1[j].T + para[2] * np.mat([1])
        elif model == 'H1*':
            if p!=1:
                delta = para[0] * para[1].T * np.hstack([y_1[j],np.mat([1])]).T + para[2] * dy[j:(j+p-1)].flatten().T
            else:
                delta = para[0] * para[1].T * np.hstack([y_1[j],np.mat([1])]).T
        elif model == 'H2':
            if p!=1:
                delta = para[0] * para[1].T * y_1[j].T + para[2] * dy[j:(j+p-1)].flatten().T
            else:
                delta = para[0] * para[1].T * y_1[j].T
        else:
            print('Errrrror')
            break
        dy[j+p,:] = delta.T            
        y_1[j+1] = y_1[j] + delta.T
    b = np.mat([[b1],[b2]])
    spread = b.T*lyc[p:].T
    spread_m = np.array(b.T*y_1.T).flatten()
    return spread_m,spread


def get_Estd(stock1,stock2,i,table,dy=True,D=16):
    if table.model_type.iloc[i] == 'model1':
        model = 'H2'
    elif table.model_type.iloc[i] == 'model2':
        model = 'H1*'
    elif table.model_type.iloc[i] == 'model3':
        model = 'H1'
    stock1 = stock1[i,:150]
    stock2 = stock2[i,:150]
    b1 = table.w1.iloc[i]
    b2 = table.w2.iloc[i]
    b = np.mat([[b1],[b2]])
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)#np.log(y)
    p = order_select(logy,5)
    u,A,_ = para_vecm(logy,model,p)
    constant = np.mat(A[:,0])
    A = A[:,1:]
    l = A.shape[1]
    extend = np.hstack([np.identity(l-2),np.zeros([l-2,2])])
    newA = np.vstack([A,extend])
    if not dy:
        lagy = logy[p-1:-1,:]
        for i in range(1,p):
            lagy = np.hstack([lagy,logy[p-1-i:-i-1,:]])
        MatrixA = np.mat(A)
        MatrixLagy = np.mat(lagy)
        Estimate_logy = MatrixA * MatrixLagy.T + constant
        e = logy[p:,:].T-Estimate_logy
        var = e*e.T/e.shape[1]
    else:
        var = u*u.T/u.shape[1]
    NowCoef = np.mat(np.eye(len(newA)))
    Evar = var.copy()
    for i in range(149):
        NowCoef = newA * NowCoef
        Evar = Evar + NowCoef[:2,:2]*var*NowCoef[:2,:2].T
    Evar = b.T * Evar * b
    
    return np.sqrt(Evar)

if __name__ == '__main__':
	choose_date = sys.argv[1]
	query = "select left(stime, 16) as mtimestamp, code , sum(volume * price)/(100*sum(volume)) as avg_price from s_price_tick where stime >= '"+ choose_date +" 09:00' and stime < '"+ choose_date +" 11:30' GROUP BY code, mtimestamp;"
	fin_cursor.execute(query)
	result = fin_cursor.fetchall()
	fin_db.commit()
	df = pd.DataFrame(list(result))
	df = df.pivot(index='mtimestamp', columns='code', values='avg_price')
	df = df.fillna(method='ffill')
	df = df.fillna(method='backfill')
	df.index = np.arange(0,len(df),1)
	#df = df.applymap(lambda x: x/100)
	# print(df)

	unitroot_stock = ADF.adf.drop_stationary(ADF.adf(df))  
	a = accelerate_formation.pairs_trading(unitroot_stock,1)  
	# a = accelerate_formation.pairs_trading(unitroot_stock)
	table = accelerate_formation.pairs_trading.formation_period( a )

	stock1_name = table.stock1.astype('str',copy=False)
	stock2_name = table.stock2.astype('str',copy=False)
	test_stock1 = np.array(df[stock1_name].T)
	test_stock2 = np.array(df[stock2_name].T)
	print(table)
	mean = np.zeros(len(table))
	std = np.zeros(len(table))
	for i in range(len(table)):
	    spread_m,spread = spread_mean(test_stock1,test_stock2,i,table)

	    mean[i] = np.mean(spread_m[-1:])
	    #std[i] = np.sqrt(np.mean(np.square(spread_m-spread)))
	    std[i] = get_Estd(test_stock1,test_stock2,i,table)
	table['e_mu'] = mean
	table['e_stdev'] = std	


	print(table)
	# if not table.empty:
	# 	table = table.drop(["skewness"],axis=1)
	# 	table["f_date"] = datetime.strptime(choose_date,'%Y-%m-%d')
        
		#table.to_sql("pairs", index=False,con = sqlEngine, if_exists = 'append', chunksize = 1000)
