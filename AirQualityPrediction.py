import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
import pandas.util.testing as tm
import os

path='D:\ML Projects\AirQualityUCI\AirQualityUCI.xlsx'

data=pd.read_excel(path)

print(data.head())

df=pd.DataFrame(data)

df=df.drop(df.columns[[15,16]],axis=1) #removing NaN columns

print(df)
print(df.describe()) 
print(df.info()) 

print(df.isnull().sum())  # Checking and counting for missing data points for each column
df=df.dropna() #cleaning missing values
print(df.isnull().sum())


ModeValueForColor = df['CO(GT)'].mode()[0]
print('mode value for COGT column is: ',ModeValueForColor)

x=df['CO(GT)'].value_counts()

print(x)

l=[]
for i in range(len(df.columns)):
    f=df.columns[i]
    count=0
    for j in range(len(df[f])):
        if df[f][j]==-200:
            count+=1
    l.append((f,count))
print("Values from each column that needs to be replaced with avg \n ",l)

num=df._get_numeric_data()

num[num<0]=0
print(df)
print(df.corr())
corrmat=df.corr()
top_corr_feature=corrmat.index
plt.figure(figsize=(30,20))
# to plot heat map
g=sns.heatmap(df[top_corr_feature].corr(),annot=True,cmap='viridis')
sns.pairplot(df)
df.plot(kind='scatter',x='C6H6(GT)',y='PT08.S5(O3)')


df["T"].plot.hist(bins=50)
plt.show()
#features
feature=df
feature=feature.drop('Date',axis=1)
feature=feature.drop('Time',axis=1)
feature=feature.drop('C6H6(GT)',axis=1)
feature.head()

label = df['C6H6(GT)']
label.head()

X_train,X_test,y_train,y_test = train_test_split(feature,label,test_size=.3)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

lr = LinearRegression()
lr.fit(X_train,y_train)

lr.score(X_test,y_test)

y_pred = lr.predict(X_test)
print(y_pred)
print('Coefficients: \n',lr.coef_)

# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'% r2_score(y_test, y_pred))

# the r squared value
print('R squared value: %.2f'%r2_score(y_test, y_pred))


