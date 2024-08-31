# /project_directory/pipelines/master_table_pipeline.py
import pandas as pd
from src.utils.data_preprocessing import load_data
from src.utils.feature_engineering import create_master_table

def create_master_table_pipeline():
    """Create a master table containing all necessary features for modeling."""
    
    # Load the data
    training_data = load_data('./data/featured_training_data.csv')
    test_data = load_data('./data/featured_test_data.csv')
    outlet_info = load_data('./data/outlet_info.csv')
    
    # Combine training and test data if necessary
    combined_data = pd.concat([training_data, test_data], axis=0)
    
    # Create the master table using feature engineering functions
    master_table = create_master_table(combined_data, outlet_info)
    
    # Save the master table
    master_table.to_csv('./data/master_table.csv', index=False)
    print("Master table created and saved.")

if __name__ == "__main__":
    create_master_table_pipeline()
