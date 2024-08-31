# /project_directory/src/utils/model_utils.py


from joblib import dump


def save_model(best_model, filename):
    """
    Save the trained model to a file.
    
    Parameters:
    model: The model to be saved.
    filename: The name of the file to save the model to.
    """
    dump(best_model, filename)
    print(f"Model saved to {filename}")