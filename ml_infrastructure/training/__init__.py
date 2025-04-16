"""
ML Infrastructure Training Module

This module provides tools for training and serving machine learning models.
"""

from ml_infrastructure.training.train_startup_model import main as train_startup_model
from ml_infrastructure.training.serve_startup_model import start_model_server, test_prediction 