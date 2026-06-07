import os, mlflow
uri = os.getenv('MLFLOW_TRACKING_URI')
print('Tracking URI:', uri)
client = mlflow.tracking.MlflowClient()
models = client.list_registered_models()
print('Registered models count:', len(models))
for m in models:
    print('Model name:', m.name, 'versions:', len(m.latest_versions))
