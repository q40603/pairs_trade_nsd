from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew
from cost import tax, slip
from integer import num_weight
import pandas as pd
import numpy as np
import pymysql
import os
import json

# db_host = '140.113.24.5'
# db_name = 'Fintech'
# db_user = 'fintech'
# db_passwd = 'financefintech'


# fin_db = pymysql.connect(
#     host=db_host,
#     user=db_user,
#     password=db_passwd,
#     db=db_name,

# )
# fin_cursor = fin_db.cursor(pymysql.cursors.DictCursor)





def pairs(
        choose_date,
        pos,
        formate_time,
        table,
        tick_data,
        half_data,
        maxi,
        tax_cost,
        cost_gate,
        capital,
        ):

    s1 = str(table.stock1[pos])
    s2 = str(table.stock2[pos])

    tw1 = table.w1[pos]
    tw2 = table.w2[pos]
    Estd = table.Estd[pos] #table.Estd[pos]
    Emu = table.Emu[pos] #

    up_open_time = 1.5
    down_open_time = up_open_time
    stop_loss_time =  10000 #actions[table.action[pos]][1] #actions[0][1] 2.5

    up_open = Emu + Estd * up_open_time  # 上開倉門檻
    down_open = Emu - Estd * down_open_time  # 下開倉門檻
    stop_loss = Estd * stop_loss_time  # stop_loss_time  # 停損門檻
    close = Emu  # 平倉(均值)

    trade_capital = 0
    cpA, cpB = 0, 0
    trading = [0, 0, 0]
    


    trade_capital = 0
    cpA, cpB = 0, 0
    trading = [0, 0, 0]


    used_news = False
    dup_news = False




    # if s1_news.empty and s2_news.empty:
    #     used_news = False

    # # 波動太小的配對不開倉
    if up_open_time * Estd < cost_gate:
        # trade_process.append([tick_data.mtimestamp[1], "配對波動太小，不開倉"])
        # print("配對波動太小，不開倉")
        trading_profit = 0
        trade = 0
        local_profit = 0
        position = 0
        return False, local_profit, trade_capital, trade, 0, 0, 0, dup_news

    t = formate_time  # formate time

    local_profit = []

    spread = tw1 * np.log(tick_data[s1]) + tw2 * np.log(tick_data[s2])


    # M = round(1 / table.zcr[pos])  # 平均持有時間
    trade = 0  # 計算開倉次數
    break_point = 0  # 計算累積斷裂點

    position = 0  # 持倉狀態，1:多倉，0:無倉，-1:空倉，-2：強制平倉
    pos = [0, 0]
    stock1_profit = []
    stock2_profit = []


    

    for i in range(0, len(spread) - 2):
        if position == 0 and i != len(spread) - 3 and i < 40:  # 之前無開倉

            

            
            
            if  spread[i] < (close + stop_loss) and  spread[i] > up_open : # 碰到上開倉門檻且小於上停損門檻
                # 資金權重轉股票張數，並整數化

                # print(tick_data.mtimestamp[i],"碰到上開倉門檻 ,上開倉")
                w1, w2 = num_weight(tw1, tw2, tick_data[s1][i], tick_data[s2][i], maxi, capital)
                position = -1
                stock1_payoff = w1 * slip(half_data[s1][2*i+1], tw1)
                stock2_payoff = w2 * slip(half_data[s2][2*i+1], tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                cpA, cpB = stock1_payoff, stock2_payoff
                if cpA > 0 and cpB > 0:
                    trade_capital += abs(cpA) + abs(cpB)
                elif cpA > 0 and cpB < 0:
                    trade_capital += abs(cpA) + 0.9 * abs(cpB)
                elif cpA < 0 and cpB > 0:
                    trade_capital += 0.9 * abs(cpA) + abs(cpB)
                elif cpA < 0 and cpB < 0:
                    trade_capital += 0.9 * abs(cpA) + 0.9 * abs(cpB)
                    # down_open = table.mu[pos] - table.stdev[pos] * close_time
                trade += 1
                # trade_process.append([tick_data.mtimestamp[i],"碰到上開倉門檻 ,上開倉<br>",w1, w2, stock1_payoff+stock2_payoff])

            elif spread[i] > (close - stop_loss) and spread[i] < down_open:  # 碰到下開倉門檻且大於下停損門檻
                # 資金權重轉股票張數，並整數化

                # print(tick_data.mtimestamp[i],"碰到下開倉門檻 ,下開倉")
                w1, w2 = num_weight(tw1, tw2, tick_data[s1][i], tick_data[s2][i], maxi, capital)
                position = 1
                stock1_payoff = -w1 * slip(half_data[s1][2*i+1], -tw1)
                stock2_payoff = -w2 * slip(half_data[s2][2*i+1], -tw2)
                stock1_payoff, stock2_payoff = tax(
                    stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                cpA, cpB = stock1_payoff, stock2_payoff
                if cpA > 0 and cpB > 0:
                    trade_capital += abs(cpA) + abs(cpB)
                elif cpA > 0 and cpB < 0:
                    trade_capital += abs(cpA) + 0.9 * abs(cpB)
                elif cpA < 0 and cpB > 0:
                    trade_capital += 0.9 * abs(cpA) + abs(cpB)
                elif cpA < 0 and cpB < 0:
                    trade_capital += 0.9 * abs(cpA) + 0.9 * abs(cpB)
                # up_open = table.mu[pos] + table.stdev[pos] * close_time
                trade += 1
                # trade_process.append([tick_data.mtimestamp[i],"碰到下開倉門檻 ,下開倉<br>", -w1, -w2, stock1_payoff+stock2_payoff])
            else:
                position = 0
                stock1_payoff = 0
                stock2_payoff = 0
        elif position == -1:  # 之前有開空倉，平空倉      
            if (spread[i] - close) < 0:  # 空倉碰到下開倉門檻即平倉

                # print(tick_data.mtimestamp[i],"之前有開空倉，碰到均值，平倉")
                position = 666  # 平倉
                stock1_payoff = -w1 * slip(half_data[s1][2*i+1], -tw1)
                stock2_payoff = -w2 * slip(half_data[s2][2*i+1], -tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[0] += 1
                # trade_process.append([tick_data.mtimestamp[i],"碰到均值，平倉<br>",-w1, -w2, stock1_payoff+stock2_payoff])
                # down_open = table.mu[pos] - table.stdev[pos] * open_time
                # 每次交易報酬做累加(最後除以交易次數做平均)
            elif spread[i] > (close + stop_loss):  # 空倉碰到上停損門檻即平倉停損

                # print(tick_data.mtimestamp[i],"之前有開空倉，碰到上停損門檻，強制平倉")
                position = -2  # 碰到停損門檻，強制平倉
                stock1_payoff = -w1 * slip(half_data[s1][2*i+1], -tw1)
                stock2_payoff = -w2 * slip(half_data[s2][2*i+1], -tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[1] += 1
                # trade_process.append([tick_data.mtimestamp[i],"碰到上停損門檻，強制平倉<br>",-w1, -w2, stock1_payoff+stock2_payoff])
                # 每次交易報酬做累加(最後除以交易次數做平均)

            elif i == (len(spread) - 3):  # 回測結束，強制平倉
                # trade_process.append([tick_data.mtimestamp[i],"回測結束，強制平倉<br>"])
                # print(tick_data.mtimestamp[i],"回測結束，強制平倉")
                position = -4
                stock1_payoff = -w1 * slip(half_data[s1][2*i+1], -tw1)
                stock2_payoff = -w2 * slip(half_data[s2][2*i+1], -tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[2] += 1
                # trade_process.append([tick_data.mtimestamp[i],"回測結束，強制平倉<br>",-w1, -w2, stock1_payoff+stock2_payoff])
                # 每次交易報酬做累加(最後除以交易次數做平均)
            else:
                position = -1
                stock1_payoff = 0
                stock2_payoff = 0
        elif position == 1:  # 之前有開多倉，平多倉
            if (spread[i] - close) > 0:

                # print(tick_data.mtimestamp[i],"之前有開多倉，碰到均值，平倉")
                position = 666  # 平倉
                stock1_payoff = w1 * slip(half_data[s1][2*i+1], tw1)
                stock2_payoff = w2 * slip(half_data[s2][2*i+1], tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[0] += 1
                # trade_process.append([tick_data.mtimestamp[i],"碰到均值，平倉<br>",w1, w2, stock1_payoff+stock2_payoff])
                # up_open = table.mu[pos] + table.stdev[pos] * open_time
                # 每次交易報酬做累加(最後除以交易次數做平均)
            elif spread[i] < (close - stop_loss):

                # print(tick_data.mtimestamp[i],"之前有開多倉，碰到下停損門檻，強制平倉")
                position = -2  # 碰到停損門檻，強制平倉
                stock1_payoff = w1 * slip(half_data[s1][2*i+1], tw1)
                stock2_payoff = w2 * slip(half_data[s2][2*i+1], tw2)
                stock1_payoff, stock2_payoff = tax(
                    stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[1] += 1

                # trade_process.append([tick_data.mtimestamp[i],"碰到下停損門檻，強制平倉<br>", w1, w2, stock1_payoff+stock2_payoff])
                # 每次交易報酬做累加(最後除以交易次數做平均)

            elif i == (len(spread) - 3):  # 回測結束，強制平倉

                # print(tick_data.mtimestamp[i],"回測結束，強制平倉")
                position = -4
                stock1_payoff = w1 * slip(half_data[s1][2*i+1], tw1)
                stock2_payoff = w2 * slip(half_data[s2][2*i+1], tw2)
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[2] += 1

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


    local_profit = trading_profit
    # local_open_num.append(trade)
    if trade == 0:  # 如果都沒有開倉，則報酬為0
        # trade_process.append([tick_data.mtimestamp.iloc[-1],"無任何交易"])
        # print("沒有開倉")
        position = 0

    return used_news, local_profit, trade_capital, trade, position, tick_data[s1][0], tick_data[s2][0], dup_news


if __name__ == '__main__':

    min_path = "../pair_data/{}/averageprice/"
    tick_path = "../pair_data/{}/minprice/"
    table_path = "../pair_data/newstdcompare{}/"
    half_path = "../pair_data/{}_halfmin/"

    news_pr_1515 = open("./result/1515_next_half.txt","a")
    # news_pr_1520 = open("./result/1520_news_text_stop.txt","a")
    # news_pr_1525 = open("./result/1525_news_text_stop.txt","a")
    # news_pr_1530 = open("./result/1530_news_text_stop.txt","a")
    # news_pr_2020 = open("./2020_news_text.txt_1_norm","a")
    # news_pr_2025 = open("./2025_news_text.txt_1_norm","a")
    # news_pr_2030 = open("./2030_news_text.txt_1_norm","a")
    # news_pr_3030 = open("./3030_news_text.txt_1_norm","a")

    for year in range(2009, 2020):
        # query = "SELECT distinct(left(DateTime,10)) as td_date FROM Fintech.Stock_1Min_Price_Tick where DateTime >= '" + \
        #     str(year) + "-01-01 09:00' and DateTime <= '" + str(year) + "-12-31 13:30';"
        # fin_cursor.execute(query)
        # result = fin_cursor.fetchall()
        # fin_db.commit()
        td_date = [f.split('_')[0] for f in os.listdir(min_path.format(year))]
        td_date.sort()
        for choose_date in td_date:
            print(choose_date)
            if not os.path.exists(
                table_path.format(year) +
                "{}_table.csv".format(
                    choose_date.replace(
                        "-",
                        ""))):
                continue
            tick_data = pd.read_csv(
                tick_path.format(year) +
                "{}_min_stock.csv".format(
                    choose_date.replace(
                        "-",
                        "")))[
                166:].reset_index()

            half_data = pd.read_csv(
                half_path.format(year) +
                "{}_half_min.csv".format(
                    choose_date.replace(
                        "-",
                        "")))[
                332:].reset_index()

            table = pd.read_csv(
                table_path.format(year) +
                "{}_table.csv".format(
                    choose_date.replace(
                        "-",
                        "")))
            for i in range(len(table)):
                is_news, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news = pairs(
                    choose_date=choose_date,
                    pos=i,
                    formate_time=166,
                    table=table,
                    tick_data=tick_data,
                    half_data=half_data,
                    maxi=5,
                    tax_cost=0.0015,
                    cost_gate=0.0015,
                    capital=300000000
                )
                print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news, file=news_pr_1515)
                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news)
                # try:
                # is_news, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news = pairs(
                #     choose_date=choose_date,
                #     pos=i,
                #     formate_time=166,
                #     table=table,
                #     tick_data=tick_data,
                #     maxi=5,
                #     tax_cost=0.0015,
                #     cost_gate=0.0015,
                #     capital=300000000
                # )

                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news, file=news_pr_1515)

                # is_news, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news = pairs(
                #     choose_date=choose_date,
                #     pos=i,
                #     formate_time=166,
                #     table=table,
                #     tick_data=tick_data,
                #     maxi=5,
                #     tax_cost=0.0015,
                #     cost_gate=0.0020,
                #     capital=300000000
                # )
                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news, file=news_pr_1520)

                # is_news, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news = pairs(
                #     choose_date=choose_date,
                #     pos=i,
                #     formate_time=166,
                #     table=table,
                #     tick_data=tick_data,
                #     maxi=5,
                #     tax_cost=0.0015,
                #     cost_gate=0.0025,
                #     capital=300000000
                # )
                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news, file=news_pr_1525)




                # is_news, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news = pairs(
                #     choose_date=choose_date,
                #     pos=i,
                #     formate_time=166,
                #     table=table,
                #     tick_data=tick_data,
                #     maxi=5,
                #     tax_cost=0.0020,
                #     cost_gate=0.0020,
                #     capital=300000000
                # )
                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news, file=news_pr_2020)

                # is_news, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news = pairs(
                #     choose_date=choose_date,
                #     pos=i,
                #     formate_time=166,
                #     table=table,
                #     tick_data=tick_data,
                #     maxi=5,
                #     tax_cost=0.0020,
                #     cost_gate=0.0030,
                #     capital=300000000
                # )
                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news, file=news_pr_2030)

                # is_news, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news = pairs(
                #     choose_date=choose_date,
                #     pos=i,
                #     formate_time=166,
                #     table=table,
                #     tick_data=tick_data,
                #     maxi=5,
                #     tax_cost=0.0030,
                #     cost_gate=0.0030,
                #     capital=300000000
                # )
                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, dup_news, file=news_pr_3030)

