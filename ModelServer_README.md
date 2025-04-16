# ModelServer - ML Model Serving Framework

ModelServer is a component of the ml_infrastructure package that makes it easy to serve machine learning models via a REST API. It supports multiple model types including ONNX, TensorFlow, PyTorch, and Pickle-serialized models.

## Features

- Serve multiple models through a single API server
- Support for different model types (ONNX, TensorFlow, PyTorch, Pickle)
- Input validation using JSON schema
- Customizable preprocessing and postprocessing hooks
- Simple API for model predictions

## Installation

The ModelServer is part of the ml_infrastructure package. Make sure to have the necessary dependencies installed:

```bash
pip install numpy pandas flask flask-cors
```

Depending on which model types you want to serve, you'll need additional dependencies:
- For ONNX models: `pip install onnxruntime`
- For TensorFlow models: `pip install tensorflow`
- For PyTorch models: `pip install torch`

## Basic Usage

Here's a minimal example of how to use ModelServer:

```python
from ml_infrastructure.serving.model_serving import ModelServer

# Initialize the server
model_server = ModelServer(host='0.0.0.0', port=5000)

# Load a model
model_server.load_model(
    model_name="my_model",
    model_path="./models/my_model.pkl",
    model_type="pickle"
)

# Start the server
model_server.start()
```

## Advanced Usage with Preprocessing and Postprocessing

For more advanced use cases, you can add preprocessing and postprocessing hooks:

```python
from ml_infrastructure.serving.model_serving import ModelServer
import numpy as np

# Define preprocessing function
def preprocess(input_data):
    # Convert input data to the format expected by the model
    features = np.array([
        input_data['feature1'],
        input_data['feature2']
    ]).reshape(1, -1)
    return {'input': features}

# Define postprocessing function
def postprocess(output):
    # Convert model output to the format expected by the client
    prediction = float(output[0])
    return {
        'prediction': prediction,
        'class': 'positive' if prediction > 0.5 else 'negative'
    }

# Initialize the server
model_server = ModelServer(host='0.0.0.0', port=5000)

# Load a model with preprocessing and postprocessing
model_server.load_model(
    model_name="advanced_model",
    model_path="./models/advanced_model.pkl",
    model_type="pickle",
    preprocessor=preprocess,
    postprocessor=postprocess
)

# Start the server
model_server.start()
```

## Input Validation with JSON Schema

You can add input validation using JSON Schema:

```python
schema = {
    'type': 'object',
    'properties': {
        'feature1': {'type': 'number'},
        'feature2': {'type': 'number'}
    },
    'required': ['feature1', 'feature2']
}

model_server.load_model(
    model_name="validated_model",
    model_path="./models/validated_model.pkl",
    model_type="pickle",
    schema=schema
)
```

## API Endpoints

Once the server is running, the following endpoints will be available:

- `GET /models` - Lists all available models
- `GET /models/{model_name}` - Gets information about a specific model
- `POST /models/{model_name}/predict` - Makes a prediction using the specified model

## Making Predictions

You can make predictions using curl:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"feature1": 0.5, "feature2": 1.2}' http://localhost:5000/models/my_model/predict
```

Or using Python:

```python
import requests

# Prepare the data
data = {
    'feature1': 0.5,
    'feature2': 1.2
}

# Make the request
response = requests.post('http://localhost:5000/models/my_model/predict', json=data)

# Print the result
print(response.json())
```

## Complete Example

See the provided `model_server_example.py` file for a complete example of how to use ModelServer with preprocessing, postprocessing, and input validation.

To run the example:

```bash
# Start the server
python model_server_example.py

# In another terminal, test the server
python model_server_example.py --test
```

## Troubleshooting

- If you get a "Address already in use" error, try using a different port with `--port`.
- Make sure your model file exists and is of the correct type.
- Check that you have the required dependencies installed for your model type.
- If your model requires GPU, ensure that the appropriate GPU drivers and libraries are installed. 