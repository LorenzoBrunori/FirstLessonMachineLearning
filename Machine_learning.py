#predire prezzi delle case boston
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


dataset = load_boston()
X = dataset['data']
y = dataset['target']
print(X[14])
print(y[14])

model = LinearRegression()
model.fit(X, y) #addestramento
_predict = model.predict(X)

mse = 0   #mean square error
mae = 0   #mean abs error

"""
With for loop
for i, y_i in enumerate(y):
     print('Target', y[i], 'Prediction', _predict[i], 'Error', _predict[i] - y[i])
    mse += (y[i] - _predict[i]) ** 2
    mae += abs(y[i] - _predict[i])

mse = mse / len(y)
mae = mae / len(y)
"""
mse = np.mean((y - _predict) ** 2)
mae = np.mean(np.abs(y - _predict))

print('my_MSE', mse)
print('my_MAE', mae)


mse_official = mean_squared_error(y , _predict)
mae_official = mean_absolute_error(y, _predict)

print('MSE_Official', mse_official)
print('MAE_Official', mae_official)
