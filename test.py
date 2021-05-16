import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

a = pd.read_csv("test.csv")

a = a.dropna(axis=1)
print(a)
stationary_stock = np.where(a.apply(lambda x: adfuller(x)[1] > 0.05 , axis = 0 ) == False ) 
a.drop(a.columns[stationary_stock], axis=1 , inplace = True)    
print(a)

