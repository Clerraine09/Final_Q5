# /project_directory/src/models/feature_importance.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_feature_importance(best_model, feature_names):
    """
    Plot the feature importance of a given model.

    Parameters:
    model: Trained model with a feature_importances_ attribute.
    feature_names: List of feature names corresponding to the model input.
    """
    # Get feature importances from the model
    importances = best_model.feature_importances_

    # Debugging: Check lengths
    print(f"Length of feature names: {len(feature_names)}")
    print(f"Length of importances: {len(importances)}")

    # Ensure lengths match
    if len(feature_names) != len(importances):
        raise ValueError("Mismatch between feature names and importances lengths.")

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort the DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importance
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))  # Show top 10 features
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()
