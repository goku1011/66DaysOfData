import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('home-data-for-ml-course/train.csv')
test_data = pd.read_csv('home-data-for-ml-course/test.csv')

# Remove rows with missing target
train_data.dropna(subset=['SalePrice'], axis=0, inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numerical columns only
numeric_cols = [col for col in train_data.columns
                    if train_data[col].dtype in ['int64','float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=50, random_state=0)),
])

from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                                cv=5, scoring='neg_mean_absolute_error')
print("Average MAE score: ", scores.mean())
# 18311.5385

# Write a useful function
def get_score(estimators):
    my_pipeline = Pipeline(steps=[
            ('preprocessor', SimpleImputer()),
            ('model', RandomForestRegressor(n_estimators=estimators, random_state=0)),
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                    cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()

results = {}
for est in range(50, 450, 50):
    results[est] = get_score(est)

import matplotlib.pyplot as plt

plt.plot(list(results.keys()), list(results.values()))
plt.show()
