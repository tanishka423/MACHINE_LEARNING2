import pandas as pd
from  sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("FuelConsumption.csv")
dataset  = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

x = dataset.iloc[: , :3]
y = dataset.iloc[: , -1]

regressor = LinearRegression()

regressor.fit(x,y)

pickle.dump(regressor,open('model.pkl','wb'))