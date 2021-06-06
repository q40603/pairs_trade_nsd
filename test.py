from VAR import *
from Stock_data import Stock_data as sd

sd = sd()

a = sd.read_csv("test.csv")
a = a.drop_empty()
a = a.keep(16,166)
a = a.to_log()
a = a.drop_stationary()

cmp_list = a.get_company_list()
for i in range(len(cmp_list)):
    for j in range(i+1, len(cmp_list)):
        z = a[cmp_list[i],cmp_list[j]].T
        model = VAR(z)
        p = model.order_select(5)
        print(i,j,p)
        # print(a[cmp_list[i],cmp_list[j]].T.shape)
# from statsmodels.tsa.api import VAR
# import VAR as VARR

# a = pd.read_csv("test.csv")
# tmp = {}
# a = a.dropna(axis=1)
# stationary_stock = np.where(a.apply(lambda x: (adfuller(x)[1] > 0.05 ), axis = 0 ) == False ) 
# a.drop(a.columns[stationary_stock], axis=1 , inplace = True)    

# a = a.drop(index=np.arange(0,16,1)) 
# a.index = np.arange(0,len(a),1)
# a = a.iloc[0: 150,:]

# stock1 = a.iloc[:,1]
# stock2 = a.iloc[:,2]
# z = np.vstack( [stock1 , stock2] ).T
# model = VAR(z)



# print(a)
# stationary_stock = np.where(a.apply(lambda x: (adfuller(x)[1] > 0.05 ), axis = 0 ) == False ) 
# print(a.columns[stationary_stock].values)
# a.drop(a.columns[stationary_stock], axis=1 , inplace = True)    
# print(a)

