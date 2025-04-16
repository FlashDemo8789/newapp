"""
Model Serving Module

This module provides a standardized way to serve machine learning models
through HTTP APIs. It supports multiple frameworks (ONNX, TensorFlow, PyTorch)
and handles model loading, prediction, and lifecycle management.

Features:
- REST API for model predictions
- Model versioning and A/B testing
- Input validation and preprocessing
- Output postprocessing
- Monitoring and logging
"""

import os
import json
import logging
import importlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional
from datetime import datetime
from flask import Flask, request, jsonify
from waitress import serve
import threading
from ml_infrastructure.model_registry.registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # More verbose logging for debugging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
    ]
)
logger = logging.getLogger(__name__)

# Try to import ONNX Runtime - not required but preferred for performance
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. ONNX model serving will not be available.")

# Try to import TensorFlow - not required but used if available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. TensorFlow model serving will not be available.")

# Try to import PyTorch - not required but used if available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PyTorch model serving will not be available.")

class ModelInput:
    """Class representing standardized model input"""
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize model input
        
        Args:
            data: Raw input data
        """
        self.data = data
        self.preprocessed = False
        self.errors = []
    
    def validate(self, schema: Dict[str, Any]) -> bool:
        """
        Validate input against schema
        
        Args:
            schema: JSON Schema definition
            
        Returns:
            True if valid, False otherwise
        """
        # For simplicity, we're doing basic validation
        # A production system would use a proper schema validator
        
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in self.data:
                self.errors.append(f"Missing required field: {field}")
        
        if self.errors:
            return False
        return True
    
    def preprocess(self, preprocessor):
        """
        Apply preprocessing to input data
        
        Args:
            preprocessor: Preprocessing function
        """
        try:
            self.data = preprocessor(self.data)
            self.preprocessed = True
        except Exception as e:
            self.errors.append(f"Preprocessing error: {str(e)}")
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Convert input to numpy format for model inference
        
        Returns:
            Dict mapping input names to numpy arrays
        """
        result = {}
        for name, value in self.data.items():
            if isinstance(value, list):
                result[name] = np.array(value)
            elif isinstance(value, (int, float, bool)):
                result[name] = np.array([value])
            elif isinstance(value, np.ndarray):
                result[name] = value
            elif isinstance(value, dict):
                # Handle dictionary values by extracting numeric data
                try:
                    # Try to find a numeric value in the dictionary
                    numeric_value = None
                    for k, v in value.items():
                        if isinstance(v, (int, float, str)) and not isinstance(v, bool):
                            try:
                                numeric_value = float(v)
                                break
                            except (ValueError, TypeError):
                                continue
                    
                    if numeric_value is not None:
                        result[name] = np.array([numeric_value])
                    else:
                        # If no numeric value is found, use a default
                        logger.warning(f"Could not find numeric value in dictionary for '{name}', using 0.0")
                        result[name] = np.array([0.0])
                except Exception as e:
                    self.errors.append(f"Cannot convert dictionary input '{name}' to numpy array: {e}")
                    result[name] = np.array([0.0])
            else:
                # Try to convert to numpy
                try:
                    # First try to convert to float if it's a string
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    result[name] = np.array(value)
                except Exception as e:
                    self.errors.append(f"Cannot convert input '{name}' to numpy array: {e}")
                    # Use a default value as fallback
                    result[name] = np.array([0.0])
                    logger.warning(f"Using default value 0.0 for '{name}' due to conversion error")
        
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert input to pandas DataFrame
        
        Returns:
            DataFrame representation of the input
        """
        try:
            return pd.DataFrame(self.data)
        except Exception as e:
            self.errors.append(f"Cannot convert to DataFrame: {str(e)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()

class ModelOutput:
    """Class representing standardized model output"""
    
    def __init__(self, data: Any, model_name: str, model_version: str):
        """
        Initialize model output
        
        Args:
            data: Raw output data from model
            model_name: Name of the model that produced this output
            model_version: Version of the model
        """
        self.data = data
        self.model_name = model_name
        self.model_version = model_version
        self.timestamp = datetime.now().isoformat()
        self.postprocessed = False
    
    def postprocess(self, postprocessor):
        """
        Apply postprocessing to output data
        
        Args:
            postprocessor: Postprocessing function
        """
        self.data = postprocessor(self.data)
        self.postprocessed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert output to dictionary format
        
        Returns:
            Dictionary representation of the output
        """
        # Convert numpy arrays to lists
        if isinstance(self.data, np.ndarray):
            data = self.data.tolist()
        elif isinstance(self.data, pd.DataFrame):
            data = self.data.to_dict(orient='records')
        else:
            data = self.data
        
        return {
            'result': data,
            'model': {
                'name': self.model_name,
                'version': self.model_version
            },
            'timestamp': self.timestamp
        }

class ONNXModelWrapper:
    """Wrapper for ONNX models"""
    
    def __init__(self, model_path: str):
        """
        Initialize ONNX model wrapper
        
        Args:
            model_path: Path to the ONNX model file
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
        
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        
        # Get input and output names
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
    
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run prediction with ONNX model
        
        Args:
            inputs: Dictionary mapping input names to numpy arrays
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        # Ensure all required inputs are provided
        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"Missing required input: {name}")
        
        # Run inference
        results = self.session.run(self.output_names, inputs)
        
        # Map outputs to names
        return {name: result for name, result in zip(self.output_names, results)}

class TensorFlowModelWrapper:
    """Wrapper for TensorFlow models"""
    
    def __init__(self, model_path: str):
        """
        Initialize TensorFlow model wrapper
        
        Args:
            model_path: Path to the TensorFlow model directory
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        self.model_path = model_path
        self.model = tf.saved_model.load(model_path)
        
        # Get concrete function for prediction
        if hasattr(self.model, 'signatures'):
            self.predict_fn = self.model.signatures['serving_default']
        else:
            # Fallback to calling the model directly
            self.predict_fn = self.model
    
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run prediction with TensorFlow model
        
        Args:
            inputs: Dictionary mapping input names to numpy arrays
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        # Convert numpy arrays to TensorFlow tensors
        tensor_inputs = {
            name: tf.convert_to_tensor(value) for name, value in inputs.items()
        }
        
        # Run inference
        results = self.predict_fn(**tensor_inputs)
        
        # Convert TensorFlow tensors to numpy arrays
        return {
            name: tensor.numpy() for name, tensor in results.items()
        }

class PyTorchModelWrapper:
    """Wrapper for PyTorch models"""
    
    def __init__(self, model_path: str):
        """
        Initialize PyTorch model wrapper
        
        Args:
            model_path: Path to the PyTorch model file
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        self.model_path = model_path
        self.model = torch.load(model_path)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run prediction with PyTorch model
        
        Args:
            inputs: Dictionary mapping input names to numpy arrays
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        # Convert numpy arrays to PyTorch tensors
        tensor_inputs = {
            name: torch.from_numpy(value).float() for name, value in inputs.items()
        }
        
        # Run inference (without gradient calculation)
        with torch.no_grad():
            # Handle both single tensor and dictionary inputs
            if len(tensor_inputs) == 1 and 'input' in tensor_inputs:
                results = self.model(tensor_inputs['input'])
                
                # Convert results to dictionary if it's a single tensor
                if isinstance(results, torch.Tensor):
                    results = {'output': results}
            else:
                results = self.model(**tensor_inputs)
        
        # Convert PyTorch tensors to numpy arrays
        if isinstance(results, torch.Tensor):
            return {'output': results.cpu().numpy()}
        elif isinstance(results, dict):
            return {
                name: tensor.cpu().numpy() for name, tensor in results.items()
            }
        else:
            # Handle tuple/list outputs (e.g., from models with multiple outputs)
            return {f'output_{i}': tensor.cpu().numpy() for i, tensor in enumerate(results)}

class PickleModelWrapper:
    """Wrapper for pickle-serialized scikit-learn and similar models"""
    
    def __init__(self, model_path: str):
        """
        Initialize pickle model wrapper
        
        Args:
            model_path: Path to the pickle model file
        """
        import pickle
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run prediction with pickle model
        
        Args:
            inputs: Dictionary mapping input names to numpy arrays
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        # For scikit-learn style models, we typically expect a single input array
        if 'input' in inputs:
            X = inputs['input']
        elif len(inputs) == 1:
            X = next(iter(inputs.values()))
        else:
            # For models that expect a DataFrame, convert the inputs
            X = pd.DataFrame(inputs)
        
        # Check for different prediction methods
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X)
            predictions = self.model.predict(X)
            return {
                'probabilities': probas,
                'predictions': predictions
            }
        elif hasattr(self.model, 'predict'):
            predictions = self.model.predict(X)
            return {'predictions': predictions}
        else:
            # Assume the model is callable directly
            predictions = self.model(X)
            return {'predictions': predictions}

class ModelServer:
    """
    Server for model prediction services
    
    This class provides a Flask API for serving model predictions,
    with support for different model types and preprocessing/postprocessing.
    """
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 5000,
        registry_dir: str = './model_store'
    ):
        """
        Initialize model server
        
        Args:
            host: Host to bind the server
            port: Port to bind the server
            registry_dir: Model registry directory
        """
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        
        # Create or load model registry
        self.registry = ModelRegistry(registry_dir)
        
        # Dictionary mapping model_name -> (model_object, preprocessor, postprocessor, schema)
        self.loaded_models = {}
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes for the server"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'models_loaded': list(self.loaded_models.keys()),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/models', methods=['GET'])
        def list_models():
            """List available models"""
            return jsonify({
                'models': self.registry.list_models()
            })
        
        @self.app.route('/models/<model_name>/versions', methods=['GET'])
        def list_versions(model_name):
            """List available versions for a model"""
            try:
                versions = self.registry.list_versions(model_name)
                return jsonify({
                    'model': model_name,
                    'versions': versions
                })
            except ValueError as e:
                return jsonify({
                    'error': str(e)
                }), 404
        
        @self.app.route('/models/<model_name>/predict', methods=['POST'])
        def predict(model_name):
            """
            Make a prediction with the model
            
            This endpoint accepts POST requests with JSON data
            and returns model predictions.
            """
            # Load model if not already loaded
            if model_name not in self.loaded_models:
                try:
                    self.load_model(model_name)
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {e}")
                    return jsonify({
                        'error': f"Model {model_name} could not be loaded: {str(e)}"
                    }), 500
            
            # Get input data
            try:
                input_data = request.json
            except Exception as e:
                return jsonify({
                    'error': f"Invalid JSON input: {str(e)}"
                }), 400
            
            # Get the model, preprocessor, postprocessor, and schema
            model, preprocessor, postprocessor, schema = self.loaded_models[model_name]
            
            # Get model metadata
            model_metadata = next(
                (v for v in self.registry.list_versions(model_name) if v['is_active']),
                {'version_id': 'unknown'}
            )
            
            # Create model input
            model_input = ModelInput(input_data)
            
            # Validate input
            if schema and not model_input.validate(schema):
                return jsonify({
                    'error': 'Validation failed',
                    'details': model_input.errors
                }), 400
            
            # Preprocess input
            if preprocessor:
                model_input.preprocess(preprocessor)
                if model_input.errors:
                    return jsonify({
                        'error': 'Preprocessing failed',
                        'details': model_input.errors
                    }), 400
            
            # Run prediction
            try:
                # Log input data for debugging
                logger.debug(f"Input data before conversion: {json.dumps(input_data)[:1000]}")
                
                # Convert to appropriate format
                numpy_input = model_input.to_numpy()
                
                # Check for conversion errors
                if model_input.errors:
                    logger.warning(f"Input conversion warnings: {model_input.errors}")
                
                # Log the numpy input for debugging
                numpy_input_debug = {k: v.shape if isinstance(v, np.ndarray) else v for k, v in numpy_input.items()}
                logger.debug(f"Numpy input after conversion: {numpy_input_debug}")
                
                # Make prediction
                try:
                    prediction_result = model.predict(numpy_input)
                except Exception as pred_error:
                    logger.error(f"Model prediction failed: {pred_error}")
                    logger.error(f"Input that caused the error: {numpy_input_debug}")
                    raise
                
                # Create output
                output = ModelOutput(
                    prediction_result,
                    model_name,
                    model_metadata['version_id']
                )
                
                # Apply postprocessing
                if postprocessor:
                    try:
                        output.postprocess(postprocessor)
                    except Exception as post_error:
                        logger.error(f"Postprocessing error: {post_error}")
                        logger.error(f"Prediction result before postprocessing: {prediction_result}")
                        raise
                
                # Return result
                return jsonify(output.to_dict())
            
            except Exception as e:
                # Log the full exception traceback for debugging
                import traceback
                logger.error(f"Prediction error: {str(e)}")
                logger.error(f"Error traceback: {traceback.format_exc()}")
                
                # Return error response with more details
                return jsonify({
                    'error': f"Prediction failed: {str(e)}",
                    'error_type': type(e).__name__
                }), 500
    
    def load_model(
        self,
        model_name: str,
        version_id: str = None,
        preprocessor=None,
        postprocessor=None,
        schema: Dict[str, Any] = None
    ):
        """
        Load a model from the registry
        
        Args:
            model_name: Name of the model to load
            version_id: Specific version to load (if None, loads active version)
            preprocessor: Function to preprocess inputs
            postprocessor: Function to postprocess outputs
            schema: JSON Schema for input validation
        """
        # Get model from registry
        model_object = self.registry.get_model(model_name, version_id)
        
        # Determine model type and create appropriate wrapper
        if hasattr(model_object, 'predict'):
            # Generic scikit-learn style
            wrapper = model_object
        elif isinstance(model_object, str) and model_object.endswith('.onnx'):
            # ONNX model
            wrapper = ONNXModelWrapper(model_object)
        elif isinstance(model_object, str) and model_object.endswith('.pt'):
            # PyTorch model
            wrapper = PyTorchModelWrapper(model_object)
        elif isinstance(model_object, str) and os.path.isdir(model_object):
            # TensorFlow SavedModel
            wrapper = TensorFlowModelWrapper(model_object)
        else:
            # Assume pickle-serialized model
            wrapper = PickleModelWrapper(model_object)
        
        # Store model with its processors
        self.loaded_models[model_name] = (wrapper, preprocessor, postprocessor, schema)
        
        logger.info(f"Loaded model {model_name}")
    
    def start(self, threaded: bool = False):
        """
        Start the model server
        
        Args:
            threaded: Whether to run in a separate thread
        """
        if threaded:
            # Start in a separate thread
            thread = threading.Thread(target=self._run_server)
            thread.daemon = True  # Daemon thread will stop when main thread stops
            thread.start()
            return thread
        else:
            # Run in the current thread (blocking)
            self._run_server()
    
    def _run_server(self):
        """Internal method to run the server"""
        logger.info(f"Starting model server on {self.host}:{self.port}")
        serve(self.app, host=self.host, port=self.port, threads=10)
    
    def stop(self):
        """Stop the model server (only useful if running in a thread)"""
        # No direct way to stop waitress, but we can unload models
        self.loaded_models.clear()
        logger.info("Unloaded all models") 