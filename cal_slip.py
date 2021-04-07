

import sys
import statistics
import pandas as pd
f = open(sys.argv[1]).readlines()
f = [i.rstrip() for i in f]
year = list(set([i.split("-")[0] for i in f]))
year.sort()
header = ["used_news", "time", "id", "profit", "capital", "trade", "position", "s1_price", "s2_price", "dup_news"]
dt = {"used_news" : bool, "time": str, "profit": float, "trade" : int, "position" : int, 'capital': float, "dup_news" : bool} 
df = pd.read_csv(sys.argv[1], names=header, sep=" ", dtype=dt, infer_datetime_format=True)
# df["time"] = pd.to_datetime(df["time"])
header = ["df2_used_news", "df2_time", "df2_id", "df2_profit", "df2_capital", "df2_trade", "df2_position", "df2_s1_price", "df2_s2_price", "df2_dup_news"]
dt = {"df2_used_news" : bool, "df2_time": str, "df2_profit": float, "df2_trade" : int, "df2_position" : int, 'df2_capital': float, "df2_dup_news" : bool} 
df2 = pd.read_csv(sys.argv[2], names=header, sep=" ", dtype=dt, infer_datetime_format=True)
# df2["df2_time"] = pd.to_datetime(df2["df2_time"])
mer = pd.concat([df, df2], axis=1)
mer = mer[mer["trade"]>0]
mer["slip"] = mer[["profit","df2_profit", "capital"]].apply(lambda x: (x.profit - x.df2_profit)/x.capital, axis=1)
print(mer[["profit","df2_profit", "slip"]])
slip = (sum(mer["profit"]) - sum(mer["df2_profit"])) / sum(mer["capital"])
print(sum(mer["slip"])/len(mer), slip)
# exit()
#print(data)
# for y in range(2015,2020):
#     tmp = df[(df['time'].str.startswith(str(y)))]
#     tmp2 = df2[(df2['df2_time'].str.startswith(str(y)))]
#     print(tmp.reset_index())
#     print(tmp2.reset_index())
#     slip = (sum(tmp["profit"]) - sum(tmp2["df2_profit"])) / sum(tmp["capital"])
#     print("slip = {}".format(slip))
    # print(tmp.groupby("position").size())
    #print("---------- {} --------------------------".format(y))
    #print("profit = {}".format(sum(tmp["profit"])))
    #print("win rate = {}% ({} / {})".format(len(tmp[tmp["profit"]>0])/len(tmp[tmp["trade"]==1]),len(tmp[tmp["profit"]>0]), len(tmp[tmp["trade"]==1])))
    #print("nc rate = {}% ({} / {})\n".format(len(tmp[tmp["position"]==666])/len(tmp[tmp["trade"]==1]), len(tmp[tmp["position"]==666]), len(tmp[tmp["trade"]==1])))
    #print(y, sum(tmp["profit"]), len(tmp[tmp["profit"]>0]), len(tmp[tmp["trade"]==1]), len(tmp[tmp["profit"]>0])/len(tmp[tmp["trade"]==1]), len(tmp[tmp["position"]==666])/len(tmp[tmp["trade"]==1]))
# for y in year:
#     print(y, end=" ")
#     pr = sum([float(i.rstrip().split()[2]) for i in f if i.startswith(y)])
#     trade = sum([int(i.rstrip().split()[4]) for i in f if i.startswith(y)])
#     win = sum([int(i.rstrip().split()[4]) for i in f if i.startswith(y) and float(i.rstrip().split()[2])>0])
#     normal = sum([int(i.rstrip().split()[4]) for i in f if i.startswith(y) and int(i.rstrip().split()[5])==666])
#     abnormal_pr = sum([float(i.rstrip().split()[2]) for i in f if i.startswith(y) and (int(i.rstrip().split()[5])!=666) ])
#     normal_pr = sum([float(i.rstrip().split()[2]) for i in f if i.startswith(y) and (int(i.rstrip().split()[5])==666) ])
#     # tmp = .rstrip().split()
#     abs_s1s2_loss = statistics.median([abs(float(i.rstrip().split()[-1]) - float(i.rstrip().split()[-2])) for i in f if i.startswith(y) and (float(i.rstrip().split()[2])<0) ])
#     abs_s1s2_win = statistics.median([abs(float(i.rstrip().split()[-1]) - float(i.rstrip().split()[-2])) for i in f if i.startswith(y) and (float(i.rstrip().split()[2])>0) ])
#     print(pr, trade, win, win/trade, normal/trade, abnormal_pr, normal_pr, abs_s1s2_loss, abs_s1s2_win)
