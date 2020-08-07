import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

all_data=pd.read_csv('owid-covid-data.csv')
all_data.reset_index(inplace=True, drop=True)
all_data.drop(all_data[all_data['location']!='India'].index, inplace=True)
all_data.drop(all_data[all_data['total_cases'] == 0].index, inplace=True)
le = preprocessing.LabelEncoder()
all_data['date'] = pd.to_datetime(all_data['date'])
all_data['Day_num'] = le.fit_transform(all_data['date'])

y=all_data['total_cases']
x=range(0, len(y))
plt.plot(x,y,'--')
plt.title('Plot')
plt.xlabel('Days')
plt.ylabel('Total cases in India from 30.01.20 to 24.07.20')
plt.show()


features = ['total_cases', 'Day_num']
req_data=all_data[features].astype('float64')
req_data.dropna(how='any', axis=0, inplace=True)
train_data=req_data.drop(['total_cases'], axis=1)
labels = req_data['total_cases']

x_train , x_test , y_train , y_test = train_test_split(train_data , labels , test_size = 0.10)

mlp = MLPRegressor(hidden_layer_sizes=(80,80),activation='relu',solver='lbfgs',max_iter=3000, alpha=0.01)

mlp.fit(X=x_train.values.reshape(x_train.shape[0],-1), y=y_train.values.ravel())
y_pred = mlp.predict(x_test)
fifteenth=mlp.predict(np.array([198]).reshape(1,-1))
print(int(fifteenth))


data = {'Actual': y_test.astype(int), 'Predicted': y_pred.astype(int)}
comparison_df = pd.DataFrame(data)
comparison_df.index = comparison_df.index-13546
print(comparison_df.head)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 score:', np.sqrt(metrics.r2_score(y_test, y_pred)))

# plot predicted data
comparison_df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
