# MNIST Classification (WSL + GPU + TensorFlow)

This repository contains an end-to-end MNIST handwritten-digit classifier, set up for:

- **WSL2 / Ubuntu 22.04**
- **NVIDIA RTX 3090** with CUDA-enabled TensorFlow
- **VS Code** + Git + SSH
- Optional **FastAPI** + **Docker** for serving

---

## Project Structure

```text
ML_Projects/
└── mnist_classification/
    ├── src/
    │   ├── main.py            # Training entrypoint (logs metrics to JSON)
    │   ├── utils.py           # Config, logging, metrics helpers
    │   └── api_fastapi.py     # (optional) FastAPI service for inference
    ├── notebooks/             # Experiments in Jupyter
    ├── data/
    │   ├── raw/               # Original data
    │   ├── processed/         # Preprocessed datasets
    │   └── metrics/           # JSON metrics per run
    ├── models/                # Saved model weights / artifacts
    ├── reports/               # Plots, figures, notes
    ├── tests/                 # Pytest tests
    ├── config.yaml            # Training / path configuration
    ├── requirements.txt       # Python dependencies
    ├── Dockerfile             # Container for FastAPI service
    └── Makefile               # Convenience commands (train, test, serve)


More or less following this structure:
project_name/
│
├── data/                     # Raw and processed datasets
├── models/                   # Saved models (.h5, .pt, etc.)
├── notebooks/                # Jupyter notebooks for exploration
├── src/                      # Training, evaluation, inference code
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils/
│
├── api/                      # FastAPI model serving
│   └── main.py
│
├── tests/                    # Unit tests
│
├── .gitignore
├── Dockerfile
├── requirements.txt
├── README.md
└── config.yaml               # Optional configuration file




#Environment Setup
Prerequisites
WSL2 with Ubuntu 22.04
Python 3.10+
NVIDIA driver + CUDA toolkit (host)
VS Code with Remote WSL + Python extensions
Git with SSH keys configured

# Clone
git clone git@github.com:karimladha1/mnist_classification.git
cd mnist_classification
# 1 Create and activate venv (inside WSL)
python3 -m venv .venv
source .venv/bin/activate
# 2 Install dependencies
pip install -r requirements.txt

#3 GPU Support

# Setup
# Clone
git clone git@github.com:karimladha1/mnist_classification.git
cd mnist_classification

# Create and activate virtual environment venv (inside WSL)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt


# --> Training the Model
# Quick run from project root with .venv activated
# this will load the dataset, train the model, save metrics

make train
# or
python src/main.py

After each run, metrics are written to:
data/metrics/metrics_<run_name>_<timestamp>.json

You can customize training settings in config.yaml:
training:
  epochs: 5
  batch_size: 64
  validation_split: 0.1
paths:
  models_dir: "models"
  logs_dir: "logs"
  metrics_dir: "data/metrics"
  run_name: "run_1"

Testing
make test
# or
pytest

# --> Serving with FastAPI
# 1. Start FastAPI server
uvicorn api.main:app --reload
# 2. API available at:
http://127.0.0.1:8000
# 3. Example JSON request
{
    "image_base64": "<base64-encoded image>"
}


Implement src/api_fastapi.py with a FastAPI app called app.
Example:
from fastapi import FastAPI
import tensorflow as tf

app = FastAPI(title="MNIST Classifier")

# TODO: load your trained model here
model = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: dict):
    # TODO: preprocess input and run model.predict
    return {"prediction": None}

Run Locally
uvicorn src.api_fastapi:app --host 0.0.0.0 --port 8000

# --> Docker
# 1. Build Docker image
docker build -t ml_project .

# 2. Run the container

uvicorn src.api_fastapi:app --host 0.0.0.0 --port 8000
Run:
docker run -p 8000:8000 mnist-classification-api
The API will be available at http://localhost:8000.

# --> Development workflow
# 0. Create branches for new features
git checkout -b feature/new-model

Typical loop:
# 1. Make code changes
# 2. Run tests
make test

# 3. Run training / experiment
make train

# 3.5 Format code
black src
isort src
flake8 src


# 4. Commit and push changes
git add .
git commit -m "Describe your change"
git push


# --> License
TBD (MIT / Apache-2.0 / etc.)
See LICENSE file for details


