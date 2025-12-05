#import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
#processing data

rental_data = pd.read_csv('data.csv')

rental_data['rental_length_days'] = pd.to_datetime(rental_data['return_date'])-pd.to_datetime(rental_data['rental_date'])
rental_data['rental_length_days'] = rental_data['rental_length_days'].dt.days

rental_data['deleted_scenes'] = rental_data['special_features'].str.contains('Deleted Scenes').astype(int)
rental_data['behind_the_scenes'] = rental_data['special_features'].str.contains('Behind the Scenes').astype(int)

X = rental_data.drop(columns=['rental_date', 'return_date', 'special_features', 'rental_length_days','amount_2','length_2','rental_rate_2'])

y = rental_data['rental_length_days']

#training and testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

#model training
rf = RandomForestRegressor(random_state=1)
param_dist = {
'n_estimators': [ 400],
'max_depth': [ 20],
'min_samples_split': [2],
'min_samples_leaf': [1],
'max_features': [ 'sqrt']
}
grid_rf = GridSearchCV(estimator=rf, param_grid=param_dist, cv=3, n_jobs=-1, verbose=2)
grid_rf.fit(X_train, y_train)
print("Best parameters found: ", grid_rf.best_params_)
best_model = grid_rf.best_estimator_
y_pred = best_model.predict(X_test)
mse = MSE(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse}")