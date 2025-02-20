import os
import warnings
import mlflow
import mlflow.pyfunc
import subprocess
from mlflow.tracking import MlflowClient

# Suppress warnings
warnings.filterwarnings('ignore')

# MLflow Client
client = MlflowClient()

# Define MLflow parameters
MODEL_NAME_PREFIX = "hf_model"  # Prefix for registered models
EXPERIMENT_NAME = "model_security_pipeline"
MODELS_DIRECTORY = "./mlflow_models/"
os.makedirs(MODELS_DIRECTORY, exist_ok=True)

# Ensure CUDA is disabled
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Initialize MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)


# **1️⃣ Log Models & Metrics Before Registration**
def log_models_and_metrics(model_paths):
    """Logs models and metrics in MLflow before registering them."""
    for model_path in model_paths:
        model_name = os.path.basename(model_path).split(".")[0]  # Extract model name from filename
        full_model_name = f"{MODEL_NAME_PREFIX}_{model_name}"  # Append prefix for clarity

        # Start an MLflow run
        with mlflow.start_run():
            # Log some example metrics (replace with actual evaluation metrics)
            mlflow.log_metric("accuracy", 0.85)  # Example metric
            mlflow.log_metric("precision", 0.90)

            # Log the model as an artifact
            print(f"Logging model: {full_model_name}")
            mlflow.pyfunc.log_model(
                artifact_path=full_model_name,
                python_model=None,  # Placeholder for real model
                registered_model_name=full_model_name
            )
            print(f"Logged model: {full_model_name}")


# **2️⃣ Retrieve All Registered Models from MLflow**
def get_all_registered_models():
    """Fetches all registered models from MLflow and returns their URIs."""
    models_to_download = []
    
    registered_models = client.search_registered_models()
    for model in registered_models:
        model_name = model.name
        latest_version = client.get_latest_versions(model_name, stages=["Staging", "Production"])
        
        for version in latest_version:
            model_uri = version.source  # Get the model's storage location
            local_model_path = os.path.join(MODELS_DIRECTORY, f"{model_name}_v{version.version}")
            
            # Download model from MLflow
            try:
                print(f"Downloading model: {model_name}, Version: {version.version}")
                mlflow.artifacts.download_artifacts(model_uri, dst_path=local_model_path)
                models_to_download.append(local_model_path)
                print(f"Downloaded {model_name}_v{version.version} successfully")
            except Exception as ex:
                print(f"Error downloading {model_name}_v{version.version}: {ex}")
    
    return models_to_download


# **3️⃣ Scan Models for Security Issues**
def scan_models(model_paths):
    """Runs ModelScan on all downloaded models."""
    scan_results = []
    
    for model_path in model_paths:
        try:
            print(f"Scanning model: {model_path}")
            result = subprocess.run(
                ["modelscan", "-r", "json", "-p", model_path],
                capture_output=True, text=True
            )
            print(f"Scan result for {model_path}: {result.stdout}")
            scan_results.append({"model": model_path, "scan_result": result.stdout})
        except Exception as ex:
            print(f"Error scanning {model_path}: {ex}")

    return scan_results


# **Main Execution**
if __name__ == '__main__':
    try:
        print("🔹 Logging models and metrics in MLflow before registration...")
        local_models = [os.path.join(MODELS_DIRECTORY, file) for file in os.listdir(MODELS_DIRECTORY) if file.endswith(".pkl")]  # Example file format
        log_models_and_metrics(local_models)

        print("🔹 Retrieving models from MLflow Model Registry...")
        models_to_scan = get_all_registered_models()

        if models_to_scan:
            print("🔹 Downloaded all models. Proceeding with security scan...")
            scan_results = scan_models(models_to_scan)

            # Process scan results (e.g., block deployment if issues found)
            for result in scan_results:
                print(result)  # Log scan results for auditing

        else:
            print("No models found in MLflow. Skipping scan.")

    except Exception as e:
        print(f"Unexpected error: {e}")
