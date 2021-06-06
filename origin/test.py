import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from MTSA import order_select , snr , zcr , JB_VECM
# import VAR as VARR

a = pd.read_csv("../test.csv")
tmp = {}
a = a.dropna(axis=1)


a = a.drop(index=np.arange(0,16,1)) 
a.index = np.arange(0,len(a),1)
a = a.iloc[0: 150,:]
a = np.log(a)

stationary_stock = np.where(a.apply(lambda x: (adfuller(x)[1] > 0.05 ), axis = 0 ) == False ) 
a.drop(a.columns[stationary_stock], axis=1 , inplace = True)    



for i in range(a.shape[1]):
    for j in range(i+1, a.shape[1]):
        stock1 = a.iloc[:,i]
        stock2 = a.iloc[:,j]
        z = np.vstack( [stock1 , stock2] ).T
        # print(z.shape)
        model = VAR(z)
        p = order_select(z,5)
        print(i,j,p)#print(p)




# print(a)
# stationary_stock = np.where(a.apply(lambda x: (adfuller(x)[1] > 0.05 ), axis = 0 ) == False ) 
# print(a.columns[stationary_stock].values)
# a.drop(a.columns[stationary_stock], axis=1 , inplace = True)    
# print(a)

