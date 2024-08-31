# /project_directory/pipelines/data_pipeline.py

from src.utils.data_preprocessing import preprocess_data, clean_data, load_data
from src.utils.feature_engineering import create_sales_features, create_item_features, create_time_features, create_outlet_related_features
import pandas as pd

def run_data_pipeline(data_path, outlet_info_path):
    """Run the data processing and feature engineering pipeline."""
    
    # Load data
    data = load_data(data_path)
    outlet_info = load_data(outlet_info_path)
    
    # Clean data
    data = clean_data(data)
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Feature engineering
    data = create_sales_features(data)
    data = create_item_features(data)
    data = create_time_features(data)
    data = create_outlet_related_features(data, outlet_info)
    
    # Return processed data
    return data

if __name__ == "__main__":
    data_path = "./data/training_data.csv"  # example path
    outlet_info_path = "./data/outlet_info.csv"  # example path
    processed_data = run_data_pipeline(data_path, outlet_info_path)
    print("Data pipeline completed.")
