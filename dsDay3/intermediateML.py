import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('home-data-for-ml-course/train.csv')
X_test_full = pd.read_csv('home-data-for-ml-course/test.csv')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=50, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]


from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_val, y_t=y_train, y_v=y_val):
    model.fit(X_t, y_t)
    return mean_absolute_error(model.predict(X_v), y_v)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))


# Model 1 MAE: 24015
# Model 2 MAE: 23740
# Model 3 MAE: 23528
# Model 4 MAE: 24051
# Model 5 MAE: 23669


# Best model
best_model = model_3
#Fit the model to the training data
best_model.fit(X, y)
# Generate test predictions
preds_test = best_model.predict(X_test)
# Save predictions
output = pd.DataFrame({'Id': X_test.index,
                        'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
