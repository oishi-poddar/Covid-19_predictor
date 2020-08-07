import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

# Read data and preprocess
all_data = pd.read_csv('owid-covid-data.csv')
all_data.drop(all_data[all_data['location']!='India'].index, inplace=True) # keep only rows for India
all_data.drop(all_data[all_data['total_cases'] == 0].index, inplace=True) # remove rows with cases=0
le = preprocessing.LabelEncoder()
all_data['date'] = pd.to_datetime(all_data['date'])
all_data['Day_num'] = le.fit_transform(all_data['date']) # convert date to Integer
all_data.reset_index(inplace=True, drop=True) # dropping the s.no to set it to default

# plot initial cases vs date
y = all_data['total_cases']
x = range(0, len(y))
plt.plot(x,y,'--')
plt.title('Plot')
plt.xlabel('Days')
plt.ylabel('Total cases in India from 30.01.20 to 24.07.20')
plt.show()

# define features and labels
features = ['total_cases', 'Day_num']
req_data = all_data[features].astype('float64')
req_data.dropna(how='any', axis=0, inplace=True) # drop null values
train_data = req_data.drop(['total_cases'], axis=1)
labels = req_data['total_cases']

# split data
x_train , x_test , y_train , y_test = train_test_split(train_data , labels , test_size = 0.10)

# define model
mlp = MLPRegressor()

# perform GridSearch  for hyper-parameter tuning
param_list = {"max_iter": [100, 2000, 3000, 5000],
              "activation": ['relu', 'identity', 'logistic', 'tanh'],
              "solver": ['lbfgs', 'adam'],
              "hidden_layer_sizes": [(20,20), (80,80), (200,200)],
              "alpha": [0.1, 1e-5, 0.01]}

gridCV = GridSearchCV(estimator=mlp, param_grid=param_list, verbose=5, n_jobs=-1, refit=True)

grid_result = gridCV.fit(x_train, y_train)
predicted = gridCV.predict(x_test)

# predicted total cases of covid-19 for 15th Aug, 2020
fifteenth=gridCV.predict(np.array([198]).reshape(1,-1))
print(fifteenth)

# print best metrics for model
print("Best: %f usinf %s" % (grid_result.best_score_, grid_result.best_params_))
means =  grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with %r" % (mean, stdev, param))

print('MAE:', metrics.mean_absolute_error(y_test, predicted))
print('MSE:', metrics.mean_squared_error(y_test, predicted))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
print('R2 score:', np.sqrt(metrics.r2_score(y_test, predicted)))

# plot predicted data

data = {'Actual': y_test.astype(int), 'Predicted': predicted.astype(int)}
comparison_df = pd.DataFrame(data)
comparison_df.index = comparison_df.index-13546

print(comparison_df.head)
comparison_df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title("Comparison of Actual vs predicted value for Neural network")
plt.xlabel("Days since 30.01.20")
plt.ylabel("Total cases")
plt.show()
