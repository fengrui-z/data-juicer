"""
Memory Predictor with Online Learning

Predicts memory usage for operators based on sample features.
Uses online learning to adapt to changing data distributions.

Based on:
- Autothrottle: Online learning for resource prediction
- Report Section 3.3: Prediction Model
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque

from .feature_extractor import SampleFeatures, FeatureExtractor


@dataclass
class PredictionResult:
    """Result of memory prediction"""
    predicted_memory_mb: float
    confidence_lower: float  # Lower bound of confidence interval
    confidence_upper: float  # Upper bound of confidence interval
    prediction_error_history: Optional[float] = None  # Recent prediction error
    
    def get_safe_prediction(self, safety_margin: float = 0.9) -> float:
        """
        Get conservative prediction with safety margin.
        
        Uses upper confidence bound to be safe.
        """
        return self.confidence_upper / safety_margin


class MemoryPredictor:
    """
    Online learning model for memory prediction.
    
    Features:
    - Incremental learning from new observations
    - Confidence intervals for predictions
    - Automatic model retraining
    - Handles different operator types
    """
    
    def __init__(
        self,
        op_name: str,
        window_size: int = 100,
        confidence_level: float = 0.95,
        min_samples_for_prediction: int = 5,
    ):
        """
        Initialize memory predictor.
        
        Args:
            op_name: Operator name
            window_size: Number of recent samples to keep for online learning
            confidence_level: Confidence level for prediction intervals (default 95%)
            min_samples_for_prediction: Minimum samples needed before making predictions
        """
        self.op_name = op_name
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.min_samples_for_prediction = min_samples_for_prediction
        
        # Online learning data
        self.feature_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.error_history = deque(maxlen=window_size)
        
        # Model parameters (online linear regression)
        self.weights: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Statistics
        self.total_predictions = 0
        self.total_updates = 0
    
    def observe(self, features: SampleFeatures, actual_memory_mb: float):
        """
        Observe a new data point and update the model.
        
        This is the core of online learning - model adapts as new data arrives.
        
        Args:
            features: Sample features
            actual_memory_mb: Actual memory used
        """
        feature_vec = np.array(features.to_feature_vector())
        
        # Store observation
        self.feature_history.append(feature_vec)
        self.memory_history.append(actual_memory_mb)
        self.total_updates += 1
        
        # Calculate prediction error if we had a model
        if self.weights is not None:
            predicted = self._predict_from_vector(feature_vec)
            error = abs(predicted - actual_memory_mb)
            self.error_history.append(error)
        
        # Retrain model if we have enough samples
        if len(self.feature_history) >= self.min_samples_for_prediction:
            self._retrain_model()
    
    def predict(self, features: SampleFeatures) -> Optional[PredictionResult]:
        """
        Predict memory usage for given features.
        
        Args:
            features: Sample features
            
        Returns:
            PredictionResult with prediction and confidence bounds, or None if not enough data
        """
        if len(self.feature_history) < self.min_samples_for_prediction:
            return None
        
        if self.weights is None:
            return None
        
        feature_vec = np.array(features.to_feature_vector())
        predicted = self._predict_from_vector(feature_vec)
        
        # Calculate confidence interval based on recent errors
        if self.error_history:
            # Use standard deviation of recent errors
            std_error = np.std(list(self.error_history))
            # For 95% confidence, use ~2 standard deviations
            z_score = 1.96 if self.confidence_level == 0.95 else 2.58
            margin = z_score * std_error
            
            confidence_lower = max(0, predicted - margin)
            confidence_upper = predicted + margin
            avg_error = np.mean(list(self.error_history))
        else:
            # No error history yet, use conservative estimate
            confidence_lower = predicted * 0.8
            confidence_upper = predicted * 1.5
            avg_error = None
        
        self.total_predictions += 1
        
        return PredictionResult(
            predicted_memory_mb=predicted,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            prediction_error_history=avg_error,
        )
    
    def predict_batch_memory(
        self,
        sample_features: SampleFeatures,
        target_batch_size: int,
    ) -> Optional[PredictionResult]:
        """
        Predict memory for a specific batch size.
        
        Scales the prediction based on batch size.
        """
        # Scale features to target batch size
        scaled_features = SampleFeatures(**vars(sample_features))
        scale_factor = target_batch_size / sample_features.batch_size
        
        scaled_features.batch_size = target_batch_size
        if scaled_features.estimated_size_mb:
            scaled_features.estimated_size_mb *= scale_factor
        
        return self.predict(scaled_features)
    
    def recommend_batch_size(
        self,
        sample_features: SampleFeatures,
        available_memory_mb: float,
        safety_margin: float = 0.85,
    ) -> int:
        """
        Recommend safe batch size given available memory.
        
        Uses binary search to find maximum safe batch size.
        
        Args:
            sample_features: Features of a single sample
            available_memory_mb: Available memory in MB
            safety_margin: Use this fraction of available memory (default 85%)
            
        Returns:
            Recommended batch size
        """
        target_memory = available_memory_mb * safety_margin
        
        # Binary search for optimal batch size
        low, high = 1, 1000
        best_batch_size = 1
        
        for _ in range(20):  # Max 20 iterations
            mid = (low + high) // 2
            prediction = self.predict_batch_memory(sample_features, mid)
            
            if prediction is None:
                # Not enough data, return conservative estimate
                return 1
            
            predicted_mem = prediction.get_safe_prediction(safety_margin)
            
            if predicted_mem <= target_memory:
                best_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return max(1, best_batch_size)
    
    def _predict_from_vector(self, feature_vec: np.ndarray) -> float:
        """Make prediction from feature vector"""
        if self.weights is None:
            return 0.0
        
        prediction = np.dot(feature_vec, self.weights) + self.intercept
        return max(0, prediction)  # Memory can't be negative
    
    def _retrain_model(self):
        """
        Retrain the model using recent observations.
        
        Uses online linear regression for efficiency.
        """
        if len(self.feature_history) < self.min_samples_for_prediction:
            return
        
        # Convert to arrays
        X = np.array(list(self.feature_history))
        y = np.array(list(self.memory_history))
        
        try:
            # Add regularization to prevent overfitting
            lambda_reg = 0.01
            n_features = X.shape[1]
            
            # Ridge regression: (X^T X + λI)^-1 X^T y
            XtX = X.T @ X
            Xty = X.T @ y
            
            # Add regularization
            XtX_reg = XtX + lambda_reg * np.eye(n_features)
            
            # Solve for weights
            self.weights = np.linalg.solve(XtX_reg, Xty)
            
            # Calculate intercept (for better fit)
            self.intercept = np.mean(y - X @ self.weights)
            
        except np.linalg.LinAlgError:
            # Singular matrix, fall back to simple mean
            self.weights = np.zeros(X.shape[1])
            self.intercept = np.mean(y)
    
    def get_model_stats(self) -> dict:
        """Get statistics about the model"""
        stats = {
            'op_name': self.op_name,
            'total_updates': self.total_updates,
            'total_predictions': self.total_predictions,
            'samples_in_window': len(self.feature_history),
            'model_trained': self.weights is not None,
        }
        
        if self.error_history:
            stats['avg_prediction_error_mb'] = float(np.mean(list(self.error_history)))
            stats['std_prediction_error_mb'] = float(np.std(list(self.error_history)))
        
        if self.memory_history:
            stats['avg_memory_mb'] = float(np.mean(list(self.memory_history)))
            stats['peak_memory_mb'] = float(np.max(list(self.memory_history)))
        
        return stats
    
    def export_model(self) -> dict:
        """Export model parameters for serialization"""
        return {
            'op_name': self.op_name,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'intercept': self.intercept,
            'window_size': self.window_size,
            'total_updates': self.total_updates,
            'stats': self.get_model_stats(),
        }
    
    def import_model(self, model_data: dict):
        """Import model parameters"""
        self.op_name = model_data['op_name']
        if model_data['weights'] is not None:
            self.weights = np.array(model_data['weights'])
        self.intercept = model_data['intercept']
        self.total_updates = model_data.get('total_updates', 0)
