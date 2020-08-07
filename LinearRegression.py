import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# Read data and preprocess
all_data = pd.read_csv('owid-covid-data.csv')
all_data.drop(all_data[all_data['location']!='India'].index, inplace=True) # keep only rows for India
all_data.drop(all_data[all_data['total_cases'] == 0].index, inplace=True) # remove rows with cases=0
le = preprocessing.LabelEncoder()
all_data['date'] = pd.to_datetime(all_data['date'])
all_data['Day_num'] = le.fit_transform(all_data['date']) # convert date to Integer
all_data.reset_index(inplace=True, drop=True) #dropping the s.no to set it to default

# Feature selection using Pearson Correlation
plt.figure(figsize=(12,10))
cor = all_data.corr()  # finding correlation between features
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds) #plotting the correlation matrix
plt.show()
# correlation with output variable
cor_target = abs(cor["total_cases"])
# Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)
# finding correlation between 2 most likely features
print(all_data[["total_deaths","total_tests"]].corr())

# plot initial cases vs date
y = all_data['total_cases']
x = range(0, len(y))
plt.plot(x,y,'--')
plt.title('Plot')
plt.xlabel('Days')
plt.ylabel('Total cases in India from 30.01.20 to 24.07.20')
plt.show()

# define features and labels
features = ['total_cases', 'Day_num', 'total_deaths']
req_data = all_data[features].astype('float64').apply(lambda x: np.log1p(x)) # apply natural logarithm to all values
req_data.replace([np.inf, -np.inf], 0, inplace=True) # Replace infinities with 0
req_data.dropna(how='any', axis=0, inplace=True)     # drop null values
train_data = req_data.drop(['total_cases'], axis=1)  # drop the target variable so training set contains only features
labels = req_data['total_cases']                     # define target variable
print(labels.shape)  #120 rows


# split data
x_train , x_test , y_train , y_test = train_test_split(train_data , labels , test_size = 0.10)

# Create linear regression model
reg = LinearRegression()

# Train the model using the training sets
reg.fit(x_train,y_train)

# Predict on test set
y_pred = reg.predict(x_test)
y_pred = np.expm1(y_pred) # applying exponential to reverse log transformation

# Predict Covid-19 cases for 15th Aug, 2020
FifteenthAugPred = reg.predict(np.log1p(np.array([198,46359])).reshape(1,-1))
print("FifteenthAugPred:", np.expm1(FifteenthAugPred).astype(int))  # applying exponential to reverse log transformation

# find metrics
y_test = np.expm1(y_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 score:', np.sqrt(metrics.r2_score(y_test, y_pred)))

# plot predicted data
data = {'Actual': y_test, 'Predicted': y_pred}
comparison_df = pd.DataFrame(data)
comparison_df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title("Linear Regression")
plt.xlabel("Days since 30.01.20")
plt.ylabel("Total cases")
plt.show()

# Regularisation using Elastic Net It is a combination of regularization L1 and L2.
model_enet = ElasticNet(normalize=True)
param_list = {"alpha":[1, 0.5, 0.1, 0.001, 0.001, 1e-4],
              "l1_ratio": [0, 0.25, 0.5, 0.75, 1]}

gridCV = GridSearchCV(estimator=model_enet, param_grid=param_list, n_jobs=-1, refit=True)

grid_result = gridCV.fit(x_train, y_train)
predicted =  gridCV.predict(x_test)
predicted = np.expm1(predicted)
print("Best: %f usinf %s" % (grid_result.best_score_, grid_result.best_params_))

# Predict Covid-19 cases for 15th Aug, 2020
FifteenthAugPred = gridCV.predict(np.log1p(np.array([198,46359])).reshape(1,-1))
print("FifteenthAugPred using Elastic net:", np.expm1(FifteenthAugPred).astype(int))  # applying exponential to reverse log transformation

# find metrics
print('MAE:', metrics.mean_absolute_error(y_test, predicted))
print('MSE:', metrics.mean_squared_error(y_test, predicted))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
print('R2 score:', np.sqrt(metrics.r2_score(y_test, predicted)))

# plot predicted data
data = {'Actual': y_test, 'Predicted': predicted}
comparison_df = pd.DataFrame(data)
print(comparison_df.head)
comparison_df.plot(kind='bar',figsize=(16,10))
plt.title("Elastic Net")
plt.xlabel("Days since 30.01.20")
plt.ylabel("Total cases")
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
