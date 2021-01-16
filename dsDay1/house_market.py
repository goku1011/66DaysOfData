import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

melb_file_path = 'melb_data.csv'

home_data = pd.read_csv(melb_file_path)
y = home_data.Price
features = ['Rooms','Landsize','Distance']
X = home_data[features]
X_train, X_val, y_train, y_val = train_test_split(X,y, random_state=1)

home_model = DecisionTreeRegressor(random_state=1)
home_model.fit(X_train, y_train)

def mae_dt(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    return mean_absolute_error(model.predict(val_X), val_y)

model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)
print('Random Forest')
print(mean_absolute_error(model.predict(X_val), y_val))

print('\n')
print('Decision Trees')
candidate_max_leaf_nodes = [100, 200, 300, 400, 500]
for i in candidate_max_leaf_nodes:
    print(mae_dt(i, X_train, X_val, y_train, y_val))
