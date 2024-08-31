import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load the trained model
model = joblib.load('final_model.joblib')

# Extract the feature names the model was trained with
model_features = model.get_booster().feature_names

# Load the results data needed for predictions
results_data = pd.read_csv('results.csv')

# Convert 'date_id' column to datetime for proper date handling
results_data['date_id'] = pd.to_datetime(results_data['date_id'])

# Load image
image_path = 'images/image1.jpg'  # Ensure this path is correct
image = Image.open(image_path)

# Define Streamlit app
def main():
    # Adding a title and description with markdown
    st.title("Sales Forecasting App")
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f6;
        }
        h1 {
            color: #4a4e69;
        }
        .stSidebar .sidebar-content {
            background-image: linear-gradient(#4a4e69, #9a8c98);
            color: white;
        }
        .css-17eq0hr {
            background-color: #f0f2f6;
        }
        .css-1v3fvcr {
            background-color: #4a4e69;
            color: white;
        }
        .css-1rhbuit-multiselectContainer {
            background-color: #4a4e69;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display image with reduced size
    st.image(image, caption='Sales Forecasting Visualization', use_column_width=False, width=500)  # Adjust width here

    # Sidebar for user input
    st.sidebar.header("Input Parameters")

    # Dropdowns and date input for selecting filters
    store = st.sidebar.selectbox("Select Store", results_data['store'].unique())
    item_dept = st.sidebar.selectbox("Select Item Department", results_data['item_dept'].unique())
    date_id = st.sidebar.date_input("Select Date", results_data['date_id'].min(), min_value=results_data['date_id'].min(), max_value=results_data['date_id'].max())

    # Filter data based on user input
    filtered_data = results_data[
        (results_data['store'] == store) &
        (results_data['item_dept'] == item_dept) &
        (results_data['date_id'] == pd.to_datetime(date_id))
    ]

    if filtered_data.empty:
        st.write("No data available for the selected inputs.")
    else:
        # Display sum of actual sales quantity
        sum_actual_quantity = filtered_data['item_qty'].sum()
        st.write(f"**Sum of Actual Sales Quantity:** {sum_actual_quantity}")

        # Prepare data for prediction by removing unnecessary columns
        input_data = filtered_data.drop(columns=['item_qty', 'store', 'item_dept', 'date_id', 'invoice_num', 'store_department_date', 'predicted_qty', 'actual_qty'])

        # Ensure input_data has the same columns as the model's training data
        missing_cols = [col for col in model_features if col not in input_data.columns]
        for col in missing_cols:
            input_data[col] = 0  # Add missing column with default value of 0

        # Keep only the columns that are in model_features
        input_data = input_data[model_features]

        # Predict sales quantity for all rows
        predicted_quantities = model.predict(input_data)

        # Calculate sum of predicted quantities
        sum_predicted_quantity = predicted_quantities.sum()
        st.write(f"**Sum of Predicted Sales Quantity:** {sum_predicted_quantity}")

if __name__ == "__main__":
    main()
