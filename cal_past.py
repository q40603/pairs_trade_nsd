import accelerate_formation
import ADF
import time
import pandas as pd
import numpy as np

import os, sys

from datetime import datetime
from sqlalchemy import create_engine
import pymysql

from vecm import para_vecm
from Matrix_function import order_select

# from sklearn import preprocessing
import torch
from ResNet.model import MSResNet
from sklearn import preprocessing

db_host = '140.113.24.5'
db_name = 'Fintech'
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


    min_path_dir = "../pair_data/{}/"
    min_path = "../pair_data/{}/averageprice/"
    tick_path = "../pair_data/{}/minprice/"
    half_path = "../pair_data/{}_halfmin/"
    table_path = "../pair_data/newstdcompare{}/"

    formate_time = 150       # 建模時間長度
    trade_time = 100         # 回測時間長度
    j = 0
    flag = 1



    corp = open("corp.txt").read().rstrip().split(",")

    
    # f = open("2019_trade_date.csv").readlines()
    # f = [i.rstrip() for i in f]
    # f = f[1:]
    
    model = MSResNet(input_channel=3, layers=[1, 1, 1, 1], num_classes=25)
    model.load_state_dict(torch.load("./ResNet/MRS_infer.pkl", map_location=torch.device('cpu')))
    model.eval()
    start = False
    print(sys.argv)
    for year in range(2004,2011):
        query = "SELECT distinct(left(DateTime,10)) as td_date FROM Fintech.Stock_1Min_Price_Tick where DateTime >= '"+ str(year) +"-01-01 09:00' and DateTime <= '"+ str(year) +"-12-31 13:30';"
        fin_cursor.execute(query)
        result = fin_cursor.fetchall()
        fin_db.commit()
        td_date = [i["td_date"] for i in result]
        print(td_date)
        os.makedirs(min_path_dir.format(year), exist_ok=True)
        os.makedirs(min_path.format(year), exist_ok=True)
        os.makedirs(tick_path.format(year), exist_ok=True)
        os.makedirs(half_path.format(year), exist_ok=True)
        os.makedirs(table_path.format(year), exist_ok=True)
        
        
        for choose_date in td_date:
            print(year, choose_date)
            # if choose_date==sys.argv[1]:
            #     start = True
            
            # if not start:
            #     continue

            fin_db.ping(reconnect = True)
            print(choose_date, end=" ")
            query = "SELECT * FROM Fintech.Stock_1Min_Price_Average where DateTime >= '"+ choose_date +" 09:00' and DateTime <= '"+ choose_date +" 13:30';"
            fin_cursor.execute(query)
            result = fin_cursor.fetchall()
            fin_db.commit()
            df = pd.DataFrame(list(result))
            df = df.drop(columns=["index","DateTime"])
            df = df[corp].astype('float')
            df.to_csv(min_path.format(year)+"{}_averagePrice_min.csv".format(choose_date.replace("-","")), index=False)
            df = df.drop(index=np.arange(0,16,1)) ; df.index = np.arange(0,len(df),1)

            avg_min = df.iloc[(trade_time * j) : (formate_time + (trade_time * j)),:] 
            avg_min.index  = np.arange(0,len(avg_min),1)
            print("save avg min data ......... done.")
            

            query = "SELECT * FROM Fintech.Stock_1Min_Price_Tick where DateTime >= '"+ choose_date +" 09:00' and DateTime <= '"+ choose_date +" 13:30';"
            fin_cursor.execute(query)
            result = fin_cursor.fetchall()
            fin_db.commit()
            df = pd.DataFrame(list(result))
            df = df.drop(columns=["index","DateTime"])
            df = df[corp].astype('float')
            df.to_csv(tick_path.format(year)+"{}_min_stock.csv".format(choose_date.replace("-","")), index=False)
            print("save tick min data ......... done.")



            query = "SELECT * FROM Fintech.Stock_30s_Price_Tick where DateTime >= '"+ choose_date +" 09:00' and DateTime <= '"+ choose_date +" 13:30';"
            fin_cursor.execute(query)
            result = fin_cursor.fetchall()
            fin_db.commit()
            halfmin = pd.DataFrame(list(result))
            halfmin = halfmin.drop(columns=["index","DateTime"])
            halfmin = halfmin[corp].astype('float')
            halfmin.to_csv(half_path.format(year)+"{}_half_min.csv".format(choose_date.replace("-","")), index=False)
            print("save half min data ......... done.")


            unitroot_stock = ADF.adf.drop_stationary(ADF.adf(avg_min))
            
            try:
                a = accelerate_formation.pairs_trading(unitroot_stock,flag)
                table = accelerate_formation.pairs_trading.formation_period( a ) 
                if table.empty:
                    continue   
                stock1_name = table.stock1.astype('str',copy=False)
                stock2_name = table.stock2.astype('str',copy=False)
                test_stock1 = np.array(df[stock1_name].T)
                test_stock2 = np.array(df[stock2_name].T)
                
                mean = np.zeros(len(table))
                std = np.zeros(len(table))
                for i in range(len(table)):
                    spread_m,spread = spread_mean(test_stock1,test_stock2,i,table)

                    mean[i] = np.mean(spread_m[-1:])
                    #std[i] = np.sqrt(np.mean(np.square(spread_m-spread)))
                    std[i] = get_Estd(test_stock1,test_stock2,i,table)
                table['Emu'] = mean
                table['Estd'] = std   

                
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
                table.to_csv(table_path.format(year)+"{}_table.csv".format(choose_date.replace("-","")), index=False)
                print("save table data ......... done.")
                print("-----------------------------------------------")    
            except :
                print("Failed to save table data")
                print("-----------------------------------------------")
                pass




    # query = "select left(stime, 19) as mtimestamp, code , sum(volume * price)/(100*sum(volume)) as avg_price from s_price_tick where stime >= '"+ choose_date +" 09:00' and stime <= '"+ choose_date +" 13:30' GROUP BY code, UNIX_TIMESTAMP(mtimestamp)DIV 30;"
    # fin_cursor.execute(query)
    # result = fin_cursor.fetchall()
    # fin_db.commit()
    # df = pd.DataFrame(list(result))
    # df = df.pivot(index='mtimestamp', columns='code', values='avg_price')
    # print(df)
    # df = df.fillna(method='ffill')
    # df = df.fillna(method='backfill')
    # df.index = pd.to_datetime(df.index)
    # halfmin = df.resample('30S').mean().copy()

    # halfmin = halfmin.reset_index()
    # halfmin.index = np.arange(0,len(halfmin),1)
    # print(halfmin)

    # model = MSResNet(input_channel=3, layers=[1, 1, 1, 1], num_classes=25)
    # model.load_state_dict(torch.load("./ResNet/MRS_infer.pkl", map_location=torch.device('cpu')))
    # model.eval()

    # from sklearn import preprocessing
    # whole_day = []
    # for pair in range(len(table)):
    #     spread = table.w1[pair] * np.log((halfmin[ str(table.stock1[pair]) ])) + table.w2[pair] * np.log(halfmin[ str(table.stock2[pair]) ])
    #     # print(spread)
    #     spread = spread[32:332].values
    #     spread = preprocessing.scale(spread)
    #     new_spread = np.zeros((3,512))
    #     new_spread[0,106:406] = spread
        
    #     mindata1 = halfmin[str(table.stock1[pair])][32:332].values
    #     mindata2 = halfmin[str(table.stock2[pair])][32:332].values

    #     # print(mindata1)
    #     # print(mindata2)
    #     mindata1 = preprocessing.scale(mindata1)
    #     mindata2 = preprocessing.scale(mindata2)
        
                
    #     new_spread[1,106:406] = mindata1
    #     new_spread[2,106:406] = mindata2

    #     whole_day.append(new_spread)

    # whole_day = np.asarray(whole_day)

    # whole_day = torch.FloatTensor(whole_day)
    # output = model(whole_day)

    # if table.shape[0] == 1:
    #     _, predicted = torch.max(output,0)
    # else:
    #     _, predicted = torch.max(output,1)
    # action_choose = predicted.cpu().numpy()
    # action_choose = action_choose.tolist()
    # table["action"] = pd.Series(action_choose)
    # print(table)  

    # if not table.empty:
    #     table = table.drop(["skewness"],axis=1)
    #     table["f_date"] = datetime.strptime(choose_date,'%Y-%m-%d')
        
    #     table.to_sql("pairs", index=False,con = sqlEngine, if_exists = 'append', chunksize = 1000)  
