# /project_directory/src/utils/feature_engineering.py

import pandas as pd

def create_sales_features(df):
    """
    Create sales-related features from historical sales data.
    """
    # lag features
    df['item_qty_lag_1'] = df.groupby(['store_department_date'])['item_qty'].shift(1)
    df['item_qty_lag_2'] = df.groupby(['store_department_date'])['item_qty'].shift(2)
    df['item_qty_lag_3'] = df.groupby(['store_department_date'])['item_qty'].shift(3)
    df['net_sales_lag_1'] = df.groupby(['store_department_date'])['net_sales'].shift(1)
    df['net_sales_lag_2'] = df.groupby(['store_department_date'])['net_sales'].shift(2)
    df['net_sales_lag_3'] = df.groupby(['store_department_date'])['net_sales'].shift(3)
    

    # Rolling window features
    df['item_qty_rolling_mean_7'] = df.groupby(['store_department_date'])['item_qty_lag_1'].shift(1).rolling(window=7).mean()
    df['net_sales_rolling_mean_7'] = df.groupby(['store_department_date'])['net_sales_lag_1'].shift(1).rolling(window=7).std()

    # Expanding window features
    df['item_qty_expanding_mean'] = df.groupby(['store_department_date'])['item_qty_lag_1'].transform(lambda x: x.expanding().mean())
    df['item_qty_expanding_sum'] = df.groupby(['store_department_date'])['item_qty_lag_1'].transform(lambda x: x.expanding().sum())
    df['net_sales_expanding_mean'] = df.groupby(['store_department_date'])['net_sales_lag_1'].transform(lambda x: x.expanding().mean())
    df['net_sales_expanding_sum'] = df.groupby(['store_department_date'])['net_sales_lag_1'].transform(lambda x: x.expanding().sum())

    # Differencing features
    #df['item_qty_diff_1'] = df.groupby(['store_department_date'])['item_qty'].diff(1)
    #df['net_sales_diff_1'] = df.groupby(['store_department_date'])['net_sales'].diff(1)

    return df

def create_item_features(df):
    """
    Create item ralted features such as price per item
    """
    df['price_per_item'] = df['net_sales_lag_1'] / df['item_qty_lag_1']

    return df

def create_time_features(df):
    """
    Create time-related features such as day of week and holidays.
    """
    df['month'] = df['date_id'].dt.month
    df['day_of_week'] = df['date_id'].dt.dayofweek 
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    holidays = pd.to_datetime(['2021-12-01', '2021-12-31'])  # Christmas where there is a peak in sales
    df['is_holiday'] = df['date_id'].isin(holidays).astype(int)

    return df


def create_outlet_related_features(df,outlet_info):
    """
    Create outlet-related features: profile, store size.
    """
    # Corrected merge function
    df = df.merge(outlet_info, on='store', how='left')

    # Convert 'profile' and 'size' to categorical type if they exist
    if 'profile' in df.columns and df['profile'].dtype == 'object':
        df['profile'] = df['profile'].astype('category')
    if 'size' in df.columns and df['size'].dtype == 'object':
        df['size'] = df['size'].astype('category')
    
    # Alternatively, use one-hot encoding for 'profile' and 'size' if needed
    df = pd.get_dummies(df, columns=['profile', 'size'], drop_first=True)

    return df



def create_master_table(df, outlet_info):
    """
    Combine all features into a master table.
    """
    df = create_sales_features(df)
    df = create_item_features(df)
    df = create_time_features(df)
    df = create_outlet_related_features(df, outlet_info)  # Pass outlet_info here
    return df
