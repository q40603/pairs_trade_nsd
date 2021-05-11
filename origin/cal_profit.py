

import sys
import statistics
import pandas as pd
f = open(sys.argv[1]).readlines()
f = [i.rstrip() for i in f]
year = list(set([i.split("-")[0] for i in f]))
year.sort()
header = ["time", "id", "profit", "capital", "trade", "position", "s1_price", "s2_price"]
dt = {"day": int, "profit": float, "trade" : int, "position" : int, 'captial': float} 
df = pd.read_csv(sys.argv[1], names=header, sep=" ", dtype=dt, infer_datetime_format=True)
df["time"] = pd.to_datetime(df["time"])
#print(data)
for y in range(2015,2020):
    tmp = df[(df['time'].dt.year == y)]
    print("---------- {} --------------------------".format(y))
    print("profit = {}".format(sum(tmp["profit"])))
    print("win rate = {}% ({} / {})".format(len(tmp[tmp["profit"]>0])/len(tmp[tmp["trade"]==1]),len(tmp[tmp["profit"]>0]), len(tmp[tmp["trade"]==1])))
    print("nc rate = {}% ({} / {})\n".format(len(tmp[tmp["position"]==666])/len(tmp[tmp["trade"]==1]), len(tmp[tmp["position"]==666]), len(tmp[tmp["trade"]==1])))
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
