import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class to train and evaluate machine learning models for price prediction"""
    
    def __init__(self, models_dir: Union[str, Path] = 'models'):
        """
        Initialize model trainer
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize scalers and models as None
        self.scaler = None
        self.model = None
        self.feature_names = None
    
    def train_classification_model(self, 
                                 X_train: np.ndarray, 
                                 y_train: np.ndarray,
                                 feature_names: List[str],
                                 model_type: str = 'random_forest',
                                 model_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Train a classification model for price direction prediction
        
        Args:
            X_train: Training features
            y_train: Training targets (0 or 1)
            feature_names: List of feature names
            model_type: Type of model to train (random_forest, gradient_boosting, logistic_regression, svm)
            model_params: Dictionary of model parameters
        """
        # Store feature names
        self.feature_names = feature_names
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create model based on type
        if model_type == 'random_forest':
            params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
            if model_params:
                params.update(model_params)
            self.model = RandomForestClassifier(**params)
        
        elif model_type == 'gradient_boosting':
            params = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
            if model_params:
                params.update(model_params)
            self.model = GradientBoostingClassifier(**params)
        
        elif model_type == 'logistic_regression':
            params = {'C': 1.0, 'random_state': 42}
            if model_params:
                params.update(model_params)
            self.model = LogisticRegression(**params)
        
        elif model_type == 'svm':
            params = {'C': 1.0, 'kernel': 'rbf', 'probability': True, 'random_state': 42}
            if model_params:
                params.update(model_params)
            self.model = SVC(**params)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        logger.info(f"Training {model_type} classification model...")
        self.model.fit(X_train_scaled, y_train)
        logger.info(f"Finished training {model_type} classification model")
    
    def evaluate_classification_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classification model on test data
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_classification_model first.")
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the trained model
        
        Returns:
            DataFrame with feature importance or None if not available
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # Check if model has feature_importances_ attribute
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            # Create DataFrame of feature importances
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        # For logistic regression, use coefficients
        elif hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_[0]
            # Create DataFrame of coefficients
            coef_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefficients
            }).sort_values('coefficient', ascending=False)
            
            return coef_df
        
        else:
            logger.warning("Feature importance not available for this model")
            return None
    
    def save_model(self, model_name: str) -> None:
        """
        Save trained model and scaler to disk
        
        Args:
            model_name: Name of the model
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Cannot save.")
        
        model_path = self.models_dir / f"{model_name}_model.pkl"
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
        feature_path = self.models_dir / f"{model_name}_features.pkl"
        
        # Save model, scaler, and feature names
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, feature_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str) -> None:
        """
        Load trained model and scaler from disk
        
        Args:
            model_name: Name of the model
        """
        model_path = self.models_dir / f"{model_name}_model.pkl"
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
        feature_path = self.models_dir / f"{model_name}_features.pkl"
        
        # Check if files exist
        if not model_path.exists() or not scaler_path.exists() or not feature_path.exists():
            raise FileNotFoundError(f"Model files for {model_name} not found")
        
        # Load model, scaler, and feature names
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(feature_path)
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with trained model
        
        Args:
            X: Features
            
        Returns:
            Predicted probabilities for positive class
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or loaded. Call train_classification_model or load_model first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Return probabilities for the positive class
        return self.model.predict_proba(X_scaled)[:, 1]