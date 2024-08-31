# /project_directory/src/models/model_training.py

from numbers import Real
import pandas as pd
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer
import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer, mean_absolute_percentage_error

def train_model(df):
    """
    Train a machine learning model to forecast sales.
    """
    # Drop unnecessary columns including 'store_department_date' which is not numeric
    X = df.drop(columns=['item_qty', 'store', 'item_dept', 'date_id', 'store_department_date', 'invoice_num'])
    y = df['item_qty']

    # Split train and validate data sets
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)

# # Define the hyperparameter space
#     param_space = {
#     'n_estimators': Integer(50, 500),
#     'max_depth': Integer(3, 10),
#     'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
#     'subsample': Real(0.6, 1.0),
#     'colsample_bytree': Real(0.6, 1.0),
#     'gamma': Real(0, 5),
#     'min_child_weight': Integer(1, 10)
#      }
    
# # Define the Bayesian search
#     bayes_search = BayesSearchCV(
#     estimator= model,
#     search_spaces=param_space,
#     n_iter=32,
#     cv=3,  # 3-fold cross-validation
#     scoring='neg_mean_absolute_percentage_error',
#     n_jobs=-1,
#     verbose=1,
#     random_state=42
#     )

#     # Finding the best model
#     bayes_search.fit(X_train, y_train)

#     # Get the best model after hyperparameter tuning
#     best_model = bayes_search.best_estimator_


    # Use eval_set for validation, and set eval_metric correctly
    eval_set = [(X_train, y_train), (X_validate, y_validate)]
    
    # Directly use eval_set without eval_metric argument
    best_model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    return best_model #, X_validate, y_validate
