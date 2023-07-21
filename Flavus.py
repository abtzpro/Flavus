import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from surprise import Dataset, Reader, SVD, accuracy
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pickle
from scipy.stats import zscore

# Obtain URLs for datasets
sales_data_url = input("Please enter the URL for your sales data: ")
org_data_url = input("Please enter the URL for your internal organization data: ")
inventory_data_url = input("Please enter the URL for your inventory data: ")

# Download data
sales_data = pd.read_csv(sales_data_url)
org_data = pd.read_csv(org_data_url)
inventory_data = pd.read_csv(inventory_data_url)

# Data preprocessing: remove outliers and handle missing values
sales_data = sales_data[(np.abs(zscore(sales_data)) < 3).all(axis=1)]
sales_data.fillna(sales_data.mean(), inplace=True)

# Basic data preprocessing
le = LabelEncoder()
scaler = StandardScaler()

# Encoding categorical features in sales data
sales_data['item'] = le.fit_transform(sales_data['item'])
sales_data['user'] = le.fit_transform(sales_data['user'])

# Scaling numerical features
sales_data['rating'] = scaler.fit_transform(sales_data['rating'].values.reshape(-1, 1))
org_data['features'] = scaler.fit_transform(org_data['features'].values.reshape(-1, 1))

# Split datasets for training and testing
sales_train, sales_test = train_test_split(sales_data, test_size=0.2, random_state=42)
org_train, org_test = train_test_split(org_data, test_size=0.2, random_state=42)

# Prepare data for Surprise library
reader = Reader(rating_scale=(sales_train['rating'].min(), sales_train['rating'].max()))
data = Dataset.load_from_df(sales_train[['user', 'item', 'rating']], reader)

# Boosting Sales: train a recommender system on sales data
trainset = data.build_full_trainset()
algo = SVD()
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

# Use the best model
algo = gs.best_estimator['rmse']
algo.fit(trainset)

# Predict on the test set
testset = list(zip(sales_test['user'].values, sales_test['item'].values, sales_test['rating'].values))
predictions = algo.test(testset)
print("Sales model RMSE: ", accuracy.rmse(predictions))
print("Sales model MAE: ", accuracy.mae(predictions))

# Internal Organization: perform clustering and regression on org data
X_train, y_train = org_train.drop('target', axis=1), org_train['target']
X_test, y_test = org_test.drop('target', axis=1), org_test['target']

kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train)
clusters = kmeans.predict(X_test)

lr = LinearRegression().fit(X_train, y_train)
preds = lr.predict(X_test)

# Evaluate the model
rmse = np.sqrt(np.mean((preds - y_test)**2))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print("Organization model RMSE: ", rmse)
print("Organization model MAE: ", mae)
print("Organization model R2 Score: ", r2)

# Inventory Visibility: perform time series forecasting on inventory data
# Handle categorical features in inventory data
inventory_data = pd.get_dummies(inventory_data, columns=['category'])

m = Prophet()
m.fit(inventory_data)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast.plot(y='yhat', figsize=(10,6), title='Inventory Forecast')

# Customer Engagement: train the recommender model on full sales data
reader = Reader(rating_scale=(sales_data['rating'].min(), sales_data['rating'].max()))
full_data = Dataset.load_from_df(sales_data[['user', 'item', 'rating']], reader)
full_trainset = full_data.build_full_trainset()
algo.fit(full_trainset)

# Save the trained model
filename = 'trained_market_model.sav'
pickle.dump(algo, open(filename, 'wb'))
print("Customer engagement model trained and saved as 'trained_market_model.sav'")
