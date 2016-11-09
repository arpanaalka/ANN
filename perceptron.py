import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
Data=pd.read_csv("dataset.csv")
Data = Data.drop(Data.index[[8293]])
ID_col=["Customer"]
target_col=["Response"]
cat_cols=['Coverage', 'Education', 'Effective To Date', 'EmploymentStatus', 'Gender',  'Location Code', 'Marital Status', 'Monthly Premium Auto', 'Policy','Policy Type', 'Renew Offer Type', 'Response', 'Sales Channel','State', 'Vehicle Class','Vehicle Size']

num_cols= list(set(list(Data.columns))-set(cat_cols)-set(ID_col)-set(target_col))
smote_cols= list(set(cat_cols)-set(target_col))
num_cat_cols = num_cols+cat_cols

#create label encoders for categorical features
for var in cat_cols:
	number = LabelEncoder()
	Data[var] = number.fit_transform(Data[var].astype('str'))
#Target variable is also a categorical so convert it
Data["Response"] = number.fit_transform(Data["Response"].astype('str'))
#print Data.head(1)


Data.to_csv('numeric.csv',columns= list(set(list(Data.columns))-set(ID_col)))

########testing and training data
x= pd.DataFrame()
features=list(set(list(Data.columns))-set(ID_col)-set(target_col))
for var in features: 
	x[var]=Data[var]
y=Data["Response"]
print y.head(1)
x = x.as_matrix(columns=None)
y = y.as_matrix(columns=None)
print "y",y
#normalizing data

for i in range(0,22) :	
	x[:,i] = (x[:,i]-min(x[:,i])) / (max(x[:,i])-min(x[:,i]))
"""
#standardization 
scaler = StandardScaler()  
scaler.fit(x)  
x = scaler.transform(x)  
# apply same transformation to test data
x = scaler.transform(x) 
"""
np.savetxt("scaled.csv", x, delimiter=",")
np.savetxt("response1.csv",y,delimiter=",")

c=len(features)
temp=0
n=0
diff=[]
W_new =[]
eta=1

Y=y
#W1=np.random.rand(1,c)
W1=np.zeros((1,c))
Y1=np.zeros(8293)
n=0
while  n<1000 :
	print n
	for i in range(0,8293):
		W2 = W1
		X = x[i,:]
		X=np.transpose(X)
		z = np.dot(W1,X)
		Y1[i] = np.sign(z)
		W1 = W2 + eta*(Y[i]-Y1[i])*X
		if all(Y1==Y)==1 :
			break
	n=n+1
	

