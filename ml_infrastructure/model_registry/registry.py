"""
Model Registry Service

This module provides a registry for managing machine learning models, their versions,
metadata, and deployment status. It enables version control, A/B testing, and 
model lifecycle management.

Usage:
    registry = ModelRegistry()
    registry.register_model("revenue_predictor", model_object, metadata={"framework": "pytorch"})
    model = registry.get_model("revenue_predictor")
"""

import os
import json
import pickle
import datetime
import uuid
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Represents a specific version of a model in the registry"""
    version_id: str
    model_path: str
    created_at: datetime.datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = False
    tags: List[str] = field(default_factory=list)
    framework: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert datetime to string for JSON serialization
        result['created_at'] = self.created_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create a ModelVersion from a dictionary"""
        # Convert string back to datetime
        if isinstance(data['created_at'], str):
            data['created_at'] = datetime.datetime.fromisoformat(data['created_at'])
        return cls(**data)

class ModelRegistry:
    """
    A registry for managing ML models, their versions, and metadata.
    
    Features:
    - Register new models and versions
    - Retrieve models by name or version
    - Manage deployment status (activate/deactivate)
    - Store and retrieve model metadata and performance metrics
    - Support A/B testing with model weights
    """
    
    def __init__(self, storage_dir: str = "./model_store"):
        """
        Initialize the model registry.
        
        Args:
            storage_dir: Directory where models and registry data will be stored
        """
        self.storage_dir = storage_dir
        self.registry_file = os.path.join(storage_dir, "registry.json")
        self.models: Dict[str, Dict[str, ModelVersion]] = {}
        self.ab_test_weights: Dict[str, Dict[str, float]] = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing registry if it exists
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load the registry from disk if it exists"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                # Populate models dictionary
                for model_name, versions_dict in data.get('models', {}).items():
                    self.models[model_name] = {}
                    for version_id, version_data in versions_dict.items():
                        self.models[model_name][version_id] = ModelVersion.from_dict(version_data)
                
                # Load A/B test weights
                self.ab_test_weights = data.get('ab_test_weights', {})
                
                logger.info(f"Loaded registry with {len(self.models)} model types")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                # Initialize with empty registry
                self.models = {}
                self.ab_test_weights = {}
    
    def _save_registry(self) -> None:
        """Save the registry to disk"""
        try:
            # Convert models to serializable format
            models_dict = {}
            for model_name, versions in self.models.items():
                models_dict[model_name] = {
                    version_id: version.to_dict() 
                    for version_id, version in versions.items()
                }
            
            data = {
                'models': models_dict,
                'ab_test_weights': self.ab_test_weights,
                'last_updated': datetime.datetime.now().isoformat()
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Registry saved to {self.registry_file}")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def register_model(
        self,
        model_name: str,
        model_object: Any,
        metadata: Dict[str, Any] = None,
        metrics: Dict[str, float] = None,
        framework: str = "unknown",
        tags: List[str] = None,
        activate: bool = False
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model type (e.g., "revenue_predictor")
            model_object: The actual model object to be saved
            metadata: Additional metadata about the model
            metrics: Performance metrics for this model version
            framework: ML framework used (pytorch, lightgbm, etc.)
            tags: List of tags for easier filtering
            activate: Whether to make this the active version
            
        Returns:
            version_id: The unique ID for this model version
        """
        # Generate a unique version ID
        version_id = str(uuid.uuid4())
        
        # Ensure model directory exists
        model_dir = os.path.join(self.storage_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model object
        model_path = os.path.join(model_dir, f"{version_id}.pkl")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_object, f)
        except Exception as e:
            logger.error(f"Error saving model object: {e}")
            raise ValueError(f"Could not save model: {e}")
        
        # Create model version entry
        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            created_at=datetime.datetime.now(),
            metrics=metrics or {},
            metadata=metadata or {},
            framework=framework,
            tags=tags or [],
            is_active=False  # Will be activated later if requested
        )
        
        # Add to registry
        if model_name not in self.models:
            self.models[model_name] = {}
        
        self.models[model_name][version_id] = version
        
        # Activate if requested
        if activate:
            self.activate_model(model_name, version_id)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered new model {model_name} version {version_id}")
        return version_id
    
    def get_model(
        self, 
        model_name: str, 
        version_id: str = None, 
        use_ab_testing: bool = False
    ) -> Any:
        """
        Retrieve a model from the registry.
        
        Args:
            model_name: Name of the model type
            version_id: Specific version to retrieve (if None, gets active version)
            use_ab_testing: If True, selects version based on A/B test weights
            
        Returns:
            The model object
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Determine which version to load
        target_version_id = version_id
        
        if target_version_id is None:
            if use_ab_testing and model_name in self.ab_test_weights:
                # Select version based on A/B test weights
                weights = self.ab_test_weights[model_name]
                versions = list(weights.keys())
                probabilities = list(weights.values())
                target_version_id = np.random.choice(versions, p=probabilities)
                logger.debug(f"A/B testing selected version {target_version_id}")
            else:
                # Find the active version
                active_versions = [
                    v_id for v_id, version in self.models[model_name].items()
                    if version.is_active
                ]
                if not active_versions:
                    raise ValueError(f"No active version found for model {model_name}")
                target_version_id = active_versions[0]
        
        if target_version_id not in self.models[model_name]:
            raise ValueError(f"Version {target_version_id} not found for model {model_name}")
        
        # Load and return the model
        model_path = self.models[model_name][target_version_id].model_path
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name} version {target_version_id}: {e}")
            raise ValueError(f"Could not load model: {e}")
    
    def activate_model(self, model_name: str, version_id: str) -> None:
        """
        Activate a specific model version (and deactivate others).
        
        Args:
            model_name: Name of the model type
            version_id: Version to activate
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version_id not in self.models[model_name]:
            raise ValueError(f"Version {version_id} not found for model {model_name}")
        
        # Deactivate all versions
        for v_id, version in self.models[model_name].items():
            version.is_active = False
        
        # Activate the specified version
        self.models[model_name][version_id].is_active = True
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Activated {model_name} version {version_id}")
    
    def setup_ab_testing(
        self, 
        model_name: str, 
        weights: Dict[str, float]
    ) -> None:
        """
        Configure A/B testing weights for model versions.
        
        Args:
            model_name: Name of the model type
            weights: Dictionary mapping version IDs to weights (must sum to 1.0)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Validate versions
        for version_id in weights.keys():
            if version_id not in self.models[model_name]:
                raise ValueError(f"Version {version_id} not found for model {model_name}")
        
        # Validate weights sum to 1.0 (with small tolerance for floating point errors)
        if not 0.99 <= sum(weights.values()) <= 1.01:
            raise ValueError(f"A/B test weights must sum to 1.0, got {sum(weights.values())}")
        
        # Store weights
        self.ab_test_weights[model_name] = weights
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Set up A/B testing for {model_name} with {len(weights)} variants")
    
    def list_models(self) -> List[str]:
        """
        List all model types in the registry.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a specific model.
        
        Args:
            model_name: Name of the model type
            
        Returns:
            List of version details
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        return [
            {
                "version_id": version_id,
                "created_at": version.created_at.isoformat(),
                "is_active": version.is_active,
                "framework": version.framework,
                "metrics": version.metrics,
                "tags": version.tags
            }
            for version_id, version in self.models[model_name].items()
        ]
    
    def delete_model(self, model_name: str) -> None:
        """
        Remove a model and all its versions from the registry.
        
        Args:
            model_name: Name of the model type to delete
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Delete model files
        for version in self.models[model_name].values():
            try:
                if os.path.exists(version.model_path):
                    os.remove(version.model_path)
            except Exception as e:
                logger.warning(f"Could not delete model file {version.model_path}: {e}")
        
        # Remove from registry
        del self.models[model_name]
        
        # Remove from A/B test weights if present
        if model_name in self.ab_test_weights:
            del self.ab_test_weights[model_name]
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Deleted model {model_name} and all its versions")
    
    def delete_version(self, model_name: str, version_id: str) -> None:
        """
        Remove a specific model version from the registry.
        
        Args:
            model_name: Name of the model type
            version_id: Version to delete
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version_id not in self.models[model_name]:
            raise ValueError(f"Version {version_id} not found for model {model_name}")
        
        # Check if this is the active version
        is_active = self.models[model_name][version_id].is_active
        
        # Delete model file
        try:
            model_path = self.models[model_name][version_id].model_path
            if os.path.exists(model_path):
                os.remove(model_path)
        except Exception as e:
            logger.warning(f"Could not delete model file: {e}")
        
        # Remove from registry
        del self.models[model_name][version_id]
        
        # Remove from A/B test weights if present
        if model_name in self.ab_test_weights and version_id in self.ab_test_weights[model_name]:
            del self.ab_test_weights[model_name][version_id]
            # Renormalize weights
            if self.ab_test_weights[model_name]:
                total = sum(self.ab_test_weights[model_name].values())
                if total > 0:
                    for v_id in self.ab_test_weights[model_name]:
                        self.ab_test_weights[model_name][v_id] /= total
        
        # If this was the active version, activate another one if available
        if is_active and self.models[model_name]:
            newest_version = max(
                self.models[model_name].values(),
                key=lambda v: v.created_at
            )
            newest_version.is_active = True
            logger.info(f"Activated {model_name} version {newest_version.version_id} to replace deleted active version")
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Deleted {model_name} version {version_id}")
    
    def update_metrics(self, model_name: str, version_id: str, metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for a model version.
        
        Args:
            model_name: Name of the model type
            version_id: Version to update
            metrics: New metrics to add/update
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version_id not in self.models[model_name]:
            raise ValueError(f"Version {version_id} not found for model {model_name}")
        
        # Update metrics
        self.models[model_name][version_id].metrics.update(metrics)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Updated metrics for {model_name} version {version_id}") 