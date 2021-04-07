import pandas as pd

import os, sys
import glob
from sklearn import preprocessing
import torch
from ResNet.model import MSResNet

if __name__ == '__main__':

    half_path = "../pair_trading_DL2/{}_halfmin/"
    table_path = "../pair_trading_DL2/newstdcompare{}/"

    formate_time = 150       # 建模時間長度
    trade_time = 100         # 回測時間長度
    j = 0
    flag = 1


    start = False
    for year in range(2015,2020):
        day = glob.glob(table_path.format(year)+"*")
        print(day)
    # for choose_date in f:
    #     pd.read_csv(half_path+"{}_half_min.csv".format(choose_date.replace("-","")), index=False)






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
