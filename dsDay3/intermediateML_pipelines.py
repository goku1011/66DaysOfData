import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('home-data-for-ml-course/train.csv')
X_test_full = pd.read_csv('home-data-for-ml-course/test.csv')

# Remove rows with missing target, separate target from predictors
X_full.dropna(subset=['SalePrice'], axis=0, inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [col for col in X_train_full.columns if
                        X_train_full[col].dtype == 'object' and
                        X_train_full[col].nunique() < 10]

# Select numerical columns
numerical_cols = [col for col in X_train_full.columns if
                        X_train_full[col].dtype in ['int64','float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

#############################
# Pipelines
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
        transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
])

# Preprocessing of training data, fit model
clf.fit(X_train, y_train)
# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)
print('MAE:', mean_absolute_error(preds, y_valid))
# 17740.2903


# Preprocessing of test data, fit model
preds_test = clf.predict(X_test)
output = pd.DataFrame({'Id':X_test.index,
                        'SalePrice':preds_test})
output.to_csv('submission_pipeline.csv', index=False)
