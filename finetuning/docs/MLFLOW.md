# MLflow Model Versioning and Registry

Training runs log models to MLflow for versioning and registry.

## Tracking URI

- **Local (default):** `mlruns/` in repo root. Set `mlflow_tracking_uri: mlruns` in config.
- **Remote:** Set `MLFLOW_TRACKING_URI` env or override in config, e.g. `mlflow_tracking_uri: http://mlflow-server:5000`

## Experiment and registry

- **Experiment:** `cld-judge-finetuning` (config: `mlflow_experiment_name`)
- **Registered model:** `cld_judge` (config: `mlflow_registry_name`)

## Loading from registry

```python
import mlflow

# List versions
client = mlflow.MlflowClient()
versions = client.search_model_versions("name='cld_judge'")

# Load latest
model_uri = f"models:/cld_judge/latest"
pipeline = mlflow.transformers.load_model(model_uri)
model = pipeline["model"]
tokenizer = pipeline["tokenizer"]
```

## Transition to production

```python
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="cld_judge",
    version=1,
    stage="Production",
)
```
