import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle

data = pd.read_csv("50_Startups.csv",na_values=['0.00'])
print(data)
print("=========================================================================")
data['R&D Spend'].fillna(data['R&D Spend'].mean(),inplace=True)
data['Marketing Spend'].fillna(data['Marketing Spend'].mean(),inplace = True)
print(data)
print("=========================================================================")

cat_cols = ['State']

X = data.drop(['Profit'],axis=1)
y = data['Profit']

print(X)


X = pd.get_dummies(X, columns = cat_cols)

print(X)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)

predicted = lr.predict(X_test)

from sklearn.metrics import mean_squared_error

print("The RMSE of Linear Model: {}".format(np.sqrt(mean_squared_error(predicted,y_test))))

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 200, random_state = 42)
rf.fit(X_train,y_train)

predicted_rf = rf.predict(X_test)

print("The RMSE of Random Forest Model: {}".format(np.sqrt(mean_squared_error(predicted_rf,y_test))))

## Creating a pickle

pickle.dump(rf, open('model.pkl','wb'))

## Loading the model 
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[ 76792.34,116980.32,48173.06,1,0,0]]))













