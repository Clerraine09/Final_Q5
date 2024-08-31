# /project_directory/src/models/model_evaluation.py


from sklearn.metrics import mean_absolute_percentage_error

def evaluate_model(best_model, X_test, y_test, master_table_test):
    """
    Evaluate the model performance using MAPE.
    """
    # Get predictions
    predictions = best_model.predict(X_test)

    # Calculate overall MAPE
    overall_mape = mean_absolute_percentage_error(y_test, predictions)
    print(f"Overall MAPE: {overall_mape:.2f}")

    # Calculate granular MAPE
    # Create a DataFrame for actual and predicted values
    results = master_table_test.copy()
    results['predicted_qty'] = predictions
    results['actual_qty'] = y_test

    # Save the results
    results.to_csv('./results.csv', index=False)
    print("Results created and saved.")
    
    # Granularity 1: Store | Department | Date
    mape_granularity1 = (
        results.groupby(['store', 'item_dept', 'date_id'])
        .apply(lambda x: mean_absolute_percentage_error(x['actual_qty'], x['predicted_qty']))
        .reset_index(name='mape')
    )
    
    # Granularity 2: Store | Date
    mape_granularity2 = (
        results.groupby(['store', 'date_id'])
        .apply(lambda x: mean_absolute_percentage_error(x['actual_qty'], x['predicted_qty']))
        .reset_index(name='mape')
    )
    
    # Print MAPE scores for granularities
    print("\nMAPE by Store | Department | Date:")
    print(mape_granularity1)
    
    print("\nMAPE by Store | Date:")
    print(mape_granularity2)

    
    return overall_mape, mape_granularity1, mape_granularity2
