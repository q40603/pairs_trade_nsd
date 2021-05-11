# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:08:26 2020

@author: allen
"""

from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew
from cost import tax , slip 
from integer import num_weight
from vecm import rank
from MTSA import fore_chow , spread_chow , order_select
import pandas as pd
import numpy as np
import pymysql
import sys

# db_host = '140.113.24.2'
# db_name = 'fintech'
# db_user = 'fintech'
# db_passwd = 'financefintech'

# fin_db = pymysql.connect(
#     host = db_host,
#     user = db_user,
#     password = db_passwd,
#     db = db_name,

# )
# fin_cursor = fin_db.cursor(pymysql.cursors.DictCursor)


def pairs(pos, formate_time, table, min_data, tick_data, maxi, tax_cost, cost_gate, capital):
    actions = [
        [1.5, 3],
        [0.5000000000002669, 2.500000000000112], 
        [0.7288428324698772, 4.0090056748083995], 
        [1.1218344155846804, 3.0000000000002496], 
        [1.2162849872773496, 7.4631043256997405], 
        [1.4751902346226717, 3.9999999999997113], 
        [1.749999999999973,  3.4999999999998117], 
        [2.086678832116794,  6.2883211678832325], 
        [2.193017888055368,  4.018753606462444], 
        [2.2499999999999822, 7.500000000000021], 
        [2.6328389830508536, 8.9762711864407], 
        [2.980046948356806,  13.515845070422579], 
        [3.2499999999999982, 5.500000000000034], 
        [3.453852327447829,  11.505617977528125], 
        [3.693027210884357,  6.0739795918367605], 
        [4.000000000000004,  12.500000000000034], 
        [4.151949541284411,  10.021788990825703], 
        [4.752819548872187,  15.016917293233117], 
        [4.8633603238866225, 7.977058029689605],
        [5.7367647058823605, 13.470588235294136],
        [6.071428571428564,  16.47435897435901],
        [6.408839779005503,  10.95488029465933],
        [7.837962962962951,  12.745370370370392],
        [8.772727272727282,  18.23295454545456],
        [9.242088607594926,  14.901898734177237],
        [100,                200]
    ]
    s1 = str(table.stock1[pos])
    s2 = str(table.stock2[pos])

    tw1 = table.w1[pos]
    tw2 = table.w2[pos]
    e_stdev = table.Estd[pos]
    e_mu = table.Emu[pos]
    up_open_time = actions[table.action[pos]][0]
    down_open_time = up_open_time
    stop_loss_time = actions[table.action[pos]][1]

    
    trade_capital = 0
    cpA,cpB = 0,0
    trading =[0,0,0]

    use_fore_lag5 = False
    use_adf = False

    trade_process = []
    # # 波動太小的配對不開倉
    # if (up_open_time + down_open_time) *e_stdev < cost_gate:
        # trade_process.append([tick_data.mtimestamp[1], "配對波動太小，不開倉"])
    #     # print("配對波動太小，不開倉")
    #     trading_profit = 0
    #     trade = 0
    #     local_profit = 0
    #     local_open_num = 0
    #     local_rt = 0
    #     local_std = 0
    #     local_skew = 0
    #     local_timetrend = 0
    #     position = 0
    #     return  {
    #         'trade_history' : trade_process , 
    #         "local_profit" : local_profit , 
    #         "local_open_num" : local_open_num,
    #         "trade_capital" :trade_capital,
    #         "local_rt" : "0.000",
    #         # "profit" : trading_profit
    #     }
    #     # return local_profit, local_open_num, trade_capital, trading, trade_process


    t = formate_time  # formate time
    # stock1_seq = min_data[s1].loc[0:t]
    # stock2_seq = min_data[s2].loc[0:t]

    local_open_num = []
    local_profit = []
    local_rt = []
    local_std = []
    local_skew = []
    local_timetrend = []
 
    spread = tw1 * np.log(tick_data[s1]) + tw2 * np.log(tick_data[s2])

    up_open = e_mu + e_stdev * up_open_time  # 上開倉門檻
    down_open = e_mu - e_stdev * down_open_time  # 下開倉門檻
    stop_loss = e_stdev * stop_loss_time  # 停損門檻
    close = e_mu  # 平倉(均值)
    # M = round(1 / table.zcr[pos])  # 平均持有時間
    trade = 0  # 計算開倉次數
    break_point = 0  # 計算累積斷裂點

    position = 0  # 持倉狀態，1:多倉，0:無倉，-1:空倉，-2：強制平倉
    pos = [0, 0]
    stock1_profit = []
    stock2_profit = []
    
    for i in range(1, len(spread) - 6):
        if position == 0 and i != len(spread) - 7:  # 之前無開倉
            if (spread[i] - up_open) * (spread[i + 1] - up_open) < 0 and spread[i + 1] < (close + stop_loss):  # 碰到上開倉門檻且小於上停損門檻
                # 資金權重轉股票張數，並整數化
                
                # print(tick_data.mtimestamp[i],"碰到上開倉門檻 ,上開倉")
                w1, w2 = num_weight(tw1, tw2, tick_data[s1][i + 1], tick_data[s2][i + 1], maxi, capital)
                position = -1
                stock1_payoff = w1 * slip(tick_data[s1][i + 1], tw1)
                stock2_payoff = w2 * slip(tick_data[s2][i + 1], tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                cpA,cpB = stock1_payoff,stock2_payoff
                if cpA > 0 and cpB > 0:
                    trade_capital += abs(cpA)+abs(cpB)
                elif cpA > 0 and cpB < 0 :
                    trade_capital += abs(cpA)+0.9*abs(cpB)
                elif cpA < 0 and cpB > 0 :
                    trade_capital += 0.9*abs(cpA)+abs(cpB)
                elif cpA < 0 and cpB < 0 :
                    trade_capital += 0.9*abs(cpA)+0.9*abs(cpB)
                    # down_open = table.mu[pos] - table.stdev[pos] * close_time
                trade += 1
                # trade_process.append([tick_data.mtimestamp[i],"碰到上開倉門檻 ,上開倉<br>",w1, w2, stock1_payoff+stock2_payoff])


            elif (spread[i] - down_open) * (spread[i + 1] - down_open) < 0 and spread[i + 1] > (close - stop_loss):  # 碰到下開倉門檻且大於下停損門檻
                # 資金權重轉股票張數，並整數化
                
                # print(tick_data.mtimestamp[i],"碰到下開倉門檻 ,下開倉")
                w1, w2 = num_weight(tw1, tw2, tick_data[s1][i + 1], tick_data[s2][i + 1], maxi, capital)
                position = 1
                stock1_payoff = -w1 * slip(tick_data[s1][i + 1], -tw1)
                stock2_payoff = -w2 * slip(tick_data[s2][i + 1], -tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                cpA,cpB = stock1_payoff,stock2_payoff
                if cpA > 0 and cpB > 0:
                    trade_capital += abs(cpA)+abs(cpB)
                elif cpA > 0 and cpB < 0 :
                    trade_capital += abs(cpA)+0.9*abs(cpB)
                elif cpA < 0 and cpB > 0 :
                    trade_capital += 0.9*abs(cpA)+abs(cpB)
                elif cpA < 0 and cpB < 0 :
                    trade_capital += 0.9*abs(cpA)+0.9*abs(cpB)
                # up_open = table.mu[pos] + table.stdev[pos] * close_time
                trade += 1
                # trade_process.append([tick_data.mtimestamp[i],"碰到下開倉門檻 ,下開倉<br>", -w1, -w2, stock1_payoff+stock2_payoff])
            else:
                position = 0
                stock1_payoff = 0
                stock2_payoff = 0
        elif position == -1:  # 之前有開空倉，平空倉
            if (spread[i] - close) * (spread[i + 1] - close) < 0:  # 空倉碰到下開倉門檻即平倉
                
                # print(tick_data.mtimestamp[i],"之前有開空倉，碰到均值，平倉")
                position = 0  # 平倉
                stock1_payoff = -w1 * slip(tick_data[s1][i + 1], -tw1)
                stock2_payoff = -w2 * slip(tick_data[s2][i + 1], -tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[0]+=1
                # trade_process.append([tick_data.mtimestamp[i],"碰到均值，平倉<br>",-w1, -w2, stock1_payoff+stock2_payoff])
                # down_open = table.mu[pos] - table.stdev[pos] * open_time
                # 每次交易報酬做累加(最後除以交易次數做平均)
            elif spread[i + 1] > (close + stop_loss):  # 空倉碰到上停損門檻即平倉停損
                
                # print(tick_data.mtimestamp[i],"之前有開空倉，碰到上停損門檻，強制平倉")
                position = -2  # 碰到停損門檻，強制平倉
                stock1_payoff = -w1 * slip(tick_data[s1][i + 1], -tw1)
                stock2_payoff = -w2 * slip(tick_data[s2][i + 1], -tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[1]+=1
                # trade_process.append([tick_data.mtimestamp[i],"碰到上停損門檻，強制平倉<br>",-w1, -w2, stock1_payoff+stock2_payoff])
                # 每次交易報酬做累加(最後除以交易次數做平均)

            elif i == (len(spread) - 7):  # 回測結束，強制平倉
                # trade_process.append([tick_data.mtimestamp[i],"回測結束，強制平倉<br>"])
                # print(tick_data.mtimestamp[i],"回測結束，強制平倉")
                position = -4
                stock1_payoff = -w1 * slip(tick_data[s1][len(tick_data) - 1], -tw1)
                stock2_payoff = -w2 * slip(tick_data[s2][len(tick_data) - 1], -tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[2]+=1
                # trade_process.append([tick_data.mtimestamp[i],"回測結束，強制平倉<br>",-w1, -w2, stock1_payoff+stock2_payoff])
                # 每次交易報酬做累加(最後除以交易次數做平均)
            else:
                position = -1
                stock1_payoff = 0
                stock2_payoff = 0
        elif position == 1:  # 之前有開多倉，平多倉
            if (spread[i] - close) * (spread[i + 1] - close) < 0:
                
                # print(tick_data.mtimestamp[i],"之前有開多倉，碰到均值，平倉")
                position = 0  # 平倉
                stock1_payoff = w1 * slip(tick_data[s1][i + 1], tw1)
                stock2_payoff = w2 * slip(tick_data[s2][i + 1], tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[0]+=1
                # trade_process.append([tick_data.mtimestamp[i],"碰到均值，平倉<br>",w1, w2, stock1_payoff+stock2_payoff])
                # up_open = table.mu[pos] + table.stdev[pos] * open_time
                # 每次交易報酬做累加(最後除以交易次數做平均)
            elif spread[i + 1] < (close - stop_loss):
                
                # print(tick_data.mtimestamp[i],"之前有開多倉，碰到下停損門檻，強制平倉")
                position = -2  # 碰到停損門檻，強制平倉
                stock1_payoff = w1 * slip(tick_data[s1][i + 1], tw1)
                stock2_payoff = w2 * slip(tick_data[s2][i + 1], tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[1]+=1

                # trade_process.append([tick_data.mtimestamp[i],"碰到下停損門檻，強制平倉<br>", w1, w2, stock1_payoff+stock2_payoff])
                # 每次交易報酬做累加(最後除以交易次數做平均)

            elif i == (len(spread) - 7):  # 回測結束，強制平倉
                
                # print(tick_data.mtimestamp[i],"回測結束，強制平倉")
                position = -4
                stock1_payoff = w1 * slip(tick_data[s1][len(tick_data) - 1], tw1)
                stock2_payoff = w2 * slip(tick_data[s2][len(tick_data) - 1], tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[2]+=1

                # trade_process.append([tick_data.mtimestamp[i],"回測結束，強制平倉<br>", w1, w2, stock1_payoff+stock2_payoff])
                # 每次交易報酬做累加(最後除以交易次數做平均)
            else:
                position = 1
                stock1_payoff = 0
                stock2_payoff = 0
        else:
            # -4: 強迫平倉 -3: 結構性斷裂平倉(for lag 5) -2:停損 666:正常平倉
            if position == -2 or position == -3 or position == -4 or position == 666:
                stock1_payoff = 0
                stock2_payoff = 0
            else:
                position = 0  # 剩下時間少於預期開倉時間，則不開倉，避免損失
                stock1_payoff = 0
                stock2_payoff = 0

        pos.append(position)
        stock1_profit.append(stock1_payoff)
        stock2_profit.append(stock2_payoff)
    trading_profit = sum(stock1_profit) + sum(stock2_profit)
    
    
    if trading_profit != 0 and position == 666:
        position = 666

    local_profit = trading_profit
    # local_open_num.append(trade)
    local_open_num = trade
    if trade == 0:  # 如果都沒有開倉，則報酬為0
        # trade_process.append([tick_data.mtimestamp.iloc[-1],"無任何交易"])
        # print("沒有開倉")
        local_rt = 0
        local_std = 0
        local_skew = 0
        local_timetrend = 0
        position = 0

    else:  # 計算平均報酬
        # spread2 = w1 * np.log(min_data[s1].iloc[0:t]) + w2 * np.log(min_data[s2].iloc[0:t])

        # x = np.arange(0, t)
        # b1, b0 = np.polyfit(x, spread2, 1)
        local_rt = trading_profit/trade_capital
        # local_std = np.std(spread2)
        # local_skew = skew(spread2)
        # local_timetrend = b1

        # trade_process.append(["總損益", trading_profit])

    # if cpA > 0 and cpB > 0:
    #     trade_capital = abs(cpA)+abs(cpB)
    # elif cpA > 0 and cpB < 0 :
    #     trade_capital = abs(cpA)+0.9*abs(cpB)
    # elif cpA < 0 and cpB > 0 :
    #     trade_capital = 0.9*abs(cpA)+abs(cpB)
    # elif cpA < 0 and cpB < 0 :
    #     trade_capital = 0.9*abs(cpA)+0.9*abs(cpB)

    return local_profit, trade_capital



path_to_data = "../pair_trading_DL2"

if __name__ == '__main__':

    table_news = pd.read_csv("../../cfda/table_news.csv")
    table_news["time"] = pd.to_datetime(table_news["time"])
    table_news["action"] = 0
    before = table_news[["stock1","stock2","time","date"]].drop_duplicates()

    profit = 0
    captial = 0

    for i in range(len(before)):
        tmp = before.iloc[i].copy()
        table = table_news[(table_news["stock1"]==tmp.stock1) & (table_news["stock2"]==tmp.stock2) & (table_news["time"]==tmp.time)].copy().reset_index()
        
        year = table.date[0].split("-")[0]
        date = table.date[0].replace("-","")
        min_data = pd.read_csv("{}/{}/minprice/{}_min_stock.csv".format(path_to_data, year, date))
        tick_data = pd.read_csv("{}/{}/averageprice/{}_averagePrice_min.csv".format(path_to_data, year, date))
        result = pairs( 
            pos = 0,
            formate_time = 166,  
            table = table , 
            min_data = min_data , 
            tick_data = tick_data ,
            maxi = 5 ,
            tax_cost = 0.000, 
            cost_gate = 0.000 , 
            capital = 300000000 
        )     
        pro, cap = result
        profit += pro
        captial += cap
        print(result)