import mlflow
mlflow.set_tracking_uri('http://158.160.2.37:5000/')
client = mlflow.tracking.MlflowClient()
artifacts = client.list_artifacts('49dbed45ad1a4e889ab467482facbf00')
for a in artifacts:
    print(a.path, a.is_dir)
