import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('home-data-for-ml-course/train.csv')
X_test_full = pd.read_csv('home-data-for-ml-course/test.csv')

# Remove rows with missing SalePrice target
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
# separate target from predictors
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# using only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

missing_val_count_by_column = X_train.isnull().sum()
print(missing_val_count_by_column[missing_val_count_by_column>0])

# Shape X_train = (1168, 37)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return mean_absolute_error(model.predict(X_valid), y_valid)


############################
# Get names of columns with missing values
missing_columns = [col for col in X_train.columns if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(missing_columns, axis=1)
reduced_X_valid = X_valid.drop(missing_columns, axis=1)

# Approach 1 - Drop columns with missing values
print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
# 17952.5914

############################
# Approach 2 - Imputation using Mean
from sklearn.impute import SimpleImputer

myimputer = SimpleImputer()
imputed_X_train = pd.DataFrame(myimputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(myimputer.transform(X_valid))
# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
# 18250.6080

############################
# Approach 2 - Imputation using Median
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.fit_transform(X_valid))
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

print("MAE (Imputation using Median):")
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)
print(mean_absolute_error(model.predict(final_X_valid), y_valid))
# 18090.3841

############################
# Pre process test Data
final_X_test = pd.DataFrame(final_imputer.transform(X_test))
preds_test = model.predict(final_X_test)

# Output
output = pd.DataFrame({'Id':X_test.index, 'SalePrice':preds_test})
output.to_csv('submission_intermediateML_missing_values.csv', index=False)
