import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.api import VAR
# import VAR as VARR

a = pd.read_csv("test.csv")
tmp = {}
a = a.dropna(axis=1)
stationary_stock = np.where(a.apply(lambda x: (adfuller(x)[1] > 0.05 ), axis = 0 ) == False ) 
a.drop(a.columns[stationary_stock], axis=1 , inplace = True)    

stock1 = a.iloc[:,1]
stock2 = a.iloc[:,2]
z = np.vstack( [stock1 , stock2] ).T




# print(a)
# stationary_stock = np.where(a.apply(lambda x: (adfuller(x)[1] > 0.05 ), axis = 0 ) == False ) 
# print(a.columns[stationary_stock].values)
# a.drop(a.columns[stationary_stock], axis=1 , inplace = True)    
# print(a)

