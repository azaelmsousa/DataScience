import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print(sk.__version__)

path_feat = "../data/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv"
path_label = "../data/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv"

# Reading data
df_feat = pd.read_csv(path_feat)
df_label = pd.read_csv(path_label)
data=pd.merge(df_feat, df_label, on=["city","year","weekofyear"]).fillna(0)
data = data.drop(['city','year','week_start_date'],axis=1)

# Preparing data
y = data['total_cases']
data = data.drop(['total_cases'],axis=1)
X = preprocessing.RobustScaler().fit_transform(data.values)

# Separating training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)

#y_train = y_train.reshape(y_train.shape[0],1)
#y_test = y_train.reshape(y_test.shape[0],1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print("="*60)
print(" "*25+"Linear Regression")
print("="*60)
# Training Linear Regression Model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Applying model
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The interception
print('Interception: \n', regr.intercept_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(y_test, y_pred,  color='blue')

plt.axis([y_test.min(),y_test.max(),y_pred.min(),y_pred.max()])

plt.show()


print("="*60)
print(" "*25+"Ridge")
print("="*60)
# Training Ridge Linear Regression Model
regr = linear_model.Ridge(alpha=.5)
regr.fit(X_train, y_train)

# Applying model
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The interception
print('Interception: \n', regr.intercept_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(y_test, y_pred,  color='blue')

plt.axis([y_test.min(),y_test.max(),y_pred.min(),y_pred.max()])

plt.show()


print("="*60)
print(" "*25+"Lasso")
print("="*60)
# Training Lasso Linear Regression Model
regr = linear_model.Lasso(alpha=0.1)
regr.fit(X_train, y_train)

# Applying model
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The interception
print('Interception: \n', regr.intercept_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(y_test, y_pred,  color='blue')

plt.axis([y_test.min(),y_test.max(),y_pred.min(),y_pred.max()])

plt.show()


print("="*60)
print(" "*25+"ElasticNet")
print("="*60)
# Training ElasticNet Regression Model
regr = linear_model.ElasticNet(random_state=0)
regr.fit(X_train, y_train)

# Applying model
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The interception
print('Interception: \n', regr.intercept_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(y_test, y_pred,  color='blue')

plt.axis([y_test.min(),y_test.max(),y_pred.min(),y_pred.max()])

plt.show()