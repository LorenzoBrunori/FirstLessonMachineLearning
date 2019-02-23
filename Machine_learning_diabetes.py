from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

dataset_diabetes = load_diabetes()

feature_names = dataset_diabetes['feature_names']

data = dataset_diabetes['data']

target = dataset_diabetes['target']

model = LinearRegression()
model.fit(data, target)
prediction = model.predict(data)

# for i, y_i in enumerate(target):
#      print('Target', target[i], 'Prediction', prediction[i], 'Error', prediction[i] - target[i])

mse = np.mean((target - prediction) ** 2)
print('Mse Value: ' , mse)
mae = np.mean(np.abs(target - prediction))
print('Mae Value: ' , mae)
mse_official = mean_squared_error(target, prediction)
print('Mse_offical Value: ' , mse_official)
mae_official = mean_absolute_error(target, prediction)
print('Mae_offical Value: ' , mae_official)



# fig, axes = plt.subplots(nrows= 1, ncols= 10)
# for ax, name in zip(axes, feature_names):
#      # ax.set(xticks= [], yticks=[])
#      ax.set_title(name)
#      ax.plot(data, target,'.',color = 'red')

x = np.linspace(0, 350)
y = x
plt.plot(target, prediction, '.', color = 'black')
plt.plot(x, y, color = 'red')
plt.show()