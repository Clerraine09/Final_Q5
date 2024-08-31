# /project_directory/src/utils/data_preprocessing.py

import pandas as pd

def load_data(file_path):
    """
    Load dataset from the specified file path.
    """
    return pd.read_csv(file_path)



def clean_data(df):
    """
    Perform data cleaning steps such as handling missing values and removing duplicates.
    Missing values are available on in Invoive number field, hence no changes done.
    """
    # Handling duplicate records
    #df = df.dropna()
    #df = df.drop_duplicates()

    # Handling outliers (e.g., using IQR method)
   # numeric_columns = df.select_dtypes(include=[np.number]).columns
    #for col in numeric_columns:
       # Q1 = df[col].quantile(0.25)
       # Q3 = df[col].quantile(0.75)
        #IQR = Q3 - Q1
        #lower_bound = Q1 - 1.5 * IQR
        #upper_bound = Q3 + 1.5 * IQR
       # df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
       # df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    return df

def preprocess_data(df):
    """
    Preprocess the data by converting data types.
    """
    
    # Print column names to debug KeyError
    print("Column names in dataframe:", df.columns.tolist())
    
    # Convert the 'date_id' column to datetime format
    df['date_id'] = pd.to_datetime(df['date_id'])
    

    return df

def create_primary_key(df):
    """
    Create a unique p.rimary key for each combination of date, store and item_dept
    """


    df['store_department_date'] = df['store'].astype(str) + '|' + df['item_dept'].astype(str) + '|' + df['date_id'].astype(str)


    return df

def group_data(df):
    """
    Group the data by date, store and date_id to be used for the daily prediction
    """

    df = df.groupby(['date_id', 'store', 'item_dept'])[['item_qty','net_sales']].sum().reset_index()

    return df

def define_target_variable(df):
    """
    Define item_qty as the target variable.
    """
    target_variable = df['item_qty']

    return target_variable



   


