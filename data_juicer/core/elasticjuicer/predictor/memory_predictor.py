"""
Memory Predictor with Online Learning

Predicts memory usage for operators based on sample features.
Uses online learning to adapt to changing data distributions.

Based on:
- Autothrottle: Online learning for resource prediction
- Report Section 3.3: Prediction Model
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np
from collections import deque

from .feature_extractor import SampleFeatures, FeatureExtractor

MODEL_SCHEMA_VERSION = 1


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
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if not 1 <= min_samples_for_prediction <= window_size:
            raise ValueError("min_samples_for_prediction must be within window_size")
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
        self.feature_mean = np.zeros(len(SampleFeatures.feature_names()))
        self.feature_scale = np.ones(len(SampleFeatures.feature_names()))

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
        if actual_memory_mb < 0:
            raise ValueError("actual_memory_mb must be non-negative")
        feature_vec = self._feature_vector(features)

        # Store observation
        self.feature_history.append(feature_vec)
        self.memory_history.append(actual_memory_mb)
        self.total_updates += 1

        # Calculate prediction error if we had a model
        if self.weights is not None:
            predicted = self._predict_from_vector(feature_vec)
            error = actual_memory_mb - predicted
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
        if self.weights is None:
            return None

        feature_vec = self._feature_vector(features)
        predicted = self._predict_from_vector(feature_vec)

        # Calculate confidence interval based on recent errors
        if self.error_history:
            absolute_errors = np.abs(np.array(self.error_history))
            margin = float(np.quantile(absolute_errors, self.confidence_level))

            confidence_lower = max(0, predicted - margin)
            confidence_upper = predicted + margin
            avg_error = float(np.mean(absolute_errors))
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
        if sample_features.batch_size < 1:
            raise ValueError("sample feature batch_size must be at least 1")
        if target_batch_size < 1:
            raise ValueError("target_batch_size must be at least 1")
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
        max_batch_size: int = 1000,
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
        if available_memory_mb <= 0:
            return 1
        if not 0 < safety_margin <= 1:
            raise ValueError("safety_margin must be in (0, 1]")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        target_memory = available_memory_mb * safety_margin

        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        best_batch_size = 1

        for _ in range(20):  # Max 20 iterations
            mid = (low + high) // 2
            prediction = self.predict_batch_memory(sample_features, mid)

            if prediction is None:
                # Not enough data, return conservative estimate
                return 1

            if prediction.confidence_upper <= target_memory:
                best_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1

        return max(1, best_batch_size)

    def _predict_from_vector(self, feature_vec: np.ndarray) -> float:
        """Make prediction from feature vector"""
        if self.weights is None:
            return 0.0

        normalized = (feature_vec - self.feature_mean) / self.feature_scale
        prediction = np.dot(normalized, self.weights) + self.intercept
        return max(0, prediction)  # Memory can't be negative

    @staticmethod
    def _feature_vector(features: SampleFeatures) -> np.ndarray:
        vector = np.asarray(features.to_feature_vector(), dtype=float)
        expected = len(SampleFeatures.feature_names())
        if vector.shape != (expected,):
            raise ValueError(f"expected {expected} features, got shape {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("features must be finite")
        return vector

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
            self.feature_mean = np.mean(X, axis=0)
            self.feature_scale = np.std(X, axis=0)
            self.feature_scale[self.feature_scale < 1e-12] = 1.0
            normalized = (X - self.feature_mean) / self.feature_scale

            # Add regularization to prevent overfitting
            lambda_reg = 0.01
            n_features = normalized.shape[1]

            # Ridge regression: (X^T X + λI)^-1 X^T y
            centered_y = y - np.mean(y)
            XtX = normalized.T @ normalized
            Xty = normalized.T @ centered_y

            # Add regularization
            XtX_reg = XtX + lambda_reg * np.eye(n_features)

            # Solve for weights
            self.weights = np.linalg.solve(XtX_reg, Xty)

            # Calculate intercept (for better fit)
            self.intercept = float(np.mean(y))

        except np.linalg.LinAlgError:
            # Singular matrix, fall back to simple mean
            self.weights = np.zeros(X.shape[1])
            self.intercept = float(np.mean(y))

    def get_model_stats(self) -> dict:
        """Get statistics about the model"""
        stats = {
            "op_name": self.op_name,
            "total_updates": self.total_updates,
            "total_predictions": self.total_predictions,
            "samples_in_window": len(self.feature_history),
            "model_trained": self.weights is not None,
        }

        if self.error_history:
            absolute_errors = np.abs(np.array(self.error_history))
            stats["avg_prediction_error_mb"] = float(np.mean(absolute_errors))
            stats["std_prediction_error_mb"] = float(np.std(list(self.error_history)))

        if self.memory_history:
            stats["avg_memory_mb"] = float(np.mean(list(self.memory_history)))
            stats["peak_memory_mb"] = float(np.max(list(self.memory_history)))

        return stats

    def export_model(self, include_history: bool = True) -> dict:
        model = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "op_name": self.op_name,
            "feature_names": SampleFeatures.feature_names(),
            "weights": self.weights.tolist() if self.weights is not None else None,
            "intercept": self.intercept,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "window_size": self.window_size,
            "confidence_level": self.confidence_level,
            "min_samples_for_prediction": self.min_samples_for_prediction,
            "total_updates": self.total_updates,
            "stats": self.get_model_stats(),
        }
        if include_history:
            model.update(
                {
                    "feature_history": [v.tolist() for v in self.feature_history],
                    "memory_history": list(self.memory_history),
                    "error_history": list(self.error_history),
                }
            )
        return model

    def import_model(self, model_data: dict):
        schema_version = model_data.get("schema_version", 0)
        if schema_version not in (0, MODEL_SCHEMA_VERSION):
            raise ValueError(f"unsupported model schema version: {schema_version}")
        expected_names = SampleFeatures.feature_names()
        feature_names = model_data.get("feature_names", expected_names)
        if feature_names != expected_names:
            raise ValueError("model feature schema does not match SampleFeatures")

        self.op_name = model_data["op_name"]
        self.window_size = int(model_data.get("window_size", self.window_size))
        self.confidence_level = float(model_data.get("confidence_level", self.confidence_level))
        self.min_samples_for_prediction = int(
            model_data.get(
                "min_samples_for_prediction",
                self.min_samples_for_prediction,
            )
        )
        if self.window_size < 1:
            raise ValueError("imported window_size must be at least 1")
        if not 0 < self.confidence_level < 1:
            raise ValueError("imported confidence_level must be in (0, 1)")
        if not 1 <= self.min_samples_for_prediction <= self.window_size:
            raise ValueError("imported min_samples_for_prediction must be within window_size")
        if model_data["weights"] is not None:
            weights = np.asarray(model_data["weights"], dtype=float)
            if weights.shape != (len(expected_names),):
                raise ValueError("model weight dimension does not match feature schema")
            self.weights = weights
        else:
            self.weights = None
        self.intercept = float(model_data["intercept"])
        self.feature_mean = np.asarray(
            model_data.get("feature_mean", np.zeros(len(expected_names))),
            dtype=float,
        )
        self.feature_scale = np.asarray(
            model_data.get("feature_scale", np.ones(len(expected_names))),
            dtype=float,
        )
        if self.feature_mean.shape != (len(expected_names),):
            raise ValueError("model feature_mean dimension does not match feature schema")
        if self.feature_scale.shape != (len(expected_names),):
            raise ValueError("model feature_scale dimension does not match feature schema")
        if np.any(self.feature_scale <= 0):
            raise ValueError("model feature_scale values must be positive")
        self.total_updates = model_data.get("total_updates", 0)

        self.feature_history = deque(
            [np.array(v) for v in model_data.get("feature_history", [])],
            maxlen=self.window_size,
        )
        self.memory_history = deque(
            model_data.get("memory_history", []),
            maxlen=self.window_size,
        )
        self.error_history = deque(
            model_data.get("error_history", []),
            maxlen=self.window_size,
        )
