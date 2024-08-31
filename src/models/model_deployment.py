# /project_directory/src/models/model_deployment.py



def deploy_model(model):
    """Deploy the trained model as a web service or using a cloud-based platform."""
    try:
        model.save_model("sales_forecasting_model.json")
        # Example of cloud platform deployment (pseudo-code, replace with actual SDK/API usage)
        # cloud_service.deploy_model("sales_forecasting_model.json")
        print("Model deployed successfully!")
    except Exception as e:
        print(f"Failed to deploy model: {e}")
