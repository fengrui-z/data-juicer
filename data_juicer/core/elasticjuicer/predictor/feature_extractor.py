"""
Feature Extractor for Memory Prediction

Extracts relevant features from data samples to predict memory usage:
- Text: length, num_tokens, special_chars
- Image: width, height, channels, format
- Video: resolution, frame_count, fps, duration
- Audio: sample_rate, duration, channels

Based on Report Section 3.3 - Prediction Model
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import re


@dataclass
class SampleFeatures:
    """Features extracted from a single sample"""
    # Common features
    batch_size: int = 1
    modality: str = "text"  # text, image, video, audio, multimodal
    
    # Text features
    text_length: Optional[int] = None
    num_tokens: Optional[int] = None
    
    # Image features  
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_channels: Optional[int] = None
    num_images: Optional[int] = 0
    
    # Video features
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    frame_count: Optional[int] = None
    fps: Optional[float] = None
    num_videos: Optional[int] = 0
    
    # Audio features
    audio_sample_rate: Optional[int] = None
    audio_duration: Optional[float] = None
    num_audios: Optional[int] = 0
    
    # Derived features
    total_pixels: Optional[int] = None  # For images/videos
    estimated_size_mb: Optional[float] = None  # Rough size estimate
    
    def to_feature_vector(self) -> List[float]:
        """Convert to numerical feature vector for ML models"""
        features = [
            float(self.batch_size),
            # Text
            float(self.text_length or 0),
            float(self.num_tokens or 0),
            # Image
            float(self.image_width or 0),
            float(self.image_height or 0),
            float(self.image_channels or 0),
            float(self.num_images or 0),
            # Video
            float(self.video_width or 0),
            float(self.video_height or 0),
            float(self.frame_count or 0),
            float(self.fps or 0),
            float(self.num_videos or 0),
            # Audio
            float(self.audio_sample_rate or 0),
            float(self.audio_duration or 0),
            float(self.num_audios or 0),
            # Derived
            float(self.total_pixels or 0),
            float(self.estimated_size_mb or 0),
        ]
        return features
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get names of features in the vector"""
        return [
            'batch_size',
            'text_length', 'num_tokens',
            'image_width', 'image_height', 'image_channels', 'num_images',
            'video_width', 'video_height', 'frame_count', 'fps', 'num_videos',
            'audio_sample_rate', 'audio_duration', 'num_audios',
            'total_pixels', 'estimated_size_mb'
        ]


class FeatureExtractor:
    """
    Extracts memory-relevant features from Data-Juicer samples.
    
    Handles different modalities and data formats.
    """
    
    def __init__(self):
        pass
    
    def extract_from_sample(self, sample: Dict[str, Any]) -> SampleFeatures:
        """
        Extract features from a single sample.
        
        Args:
            sample: Data-Juicer sample dictionary
            
        Returns:
            SampleFeatures object
        """
        features = SampleFeatures(batch_size=1)
        
        # Determine modality
        has_text = bool('text' in sample and sample['text'])
        has_images = bool('images' in sample and sample['images'])
        has_videos = bool('videos' in sample and sample['videos'])
        has_audios = bool('audios' in sample and sample['audios'])
        
        modality_count = sum([has_text, has_images, has_videos, has_audios])
        if modality_count > 1:
            features.modality = "multimodal"
        elif has_text:
            features.modality = "text"
        elif has_images:
            features.modality = "image"
        elif has_videos:
            features.modality = "video"
        elif has_audios:
            features.modality = "audio"
        
        # Extract text features
        if has_text:
            text = sample['text']
            features.text_length = len(text)
            # Simple tokenization (space-based)
            features.num_tokens = len(text.split())
        
        # Extract image features
        if has_images:
            images = sample['images']
            features.num_images = len(images) if isinstance(images, list) else 1
            # Try to get image metadata if available
            if 'image_metadata' in sample:
                meta = sample['image_metadata']
                if isinstance(meta, list) and meta:
                    meta = meta[0]  # Use first image
                features.image_width = meta.get('width')
                features.image_height = meta.get('height')
                features.image_channels = meta.get('channels', 3)
                
                if features.image_width and features.image_height:
                    features.total_pixels = features.image_width * features.image_height * features.num_images
        
        # Extract video features
        if has_videos:
            videos = sample['videos']
            features.num_videos = len(videos) if isinstance(videos, list) else 1
            # Try to get video metadata
            if 'video_metadata' in sample:
                meta = sample['video_metadata']
                if isinstance(meta, list) and meta:
                    meta = meta[0]  # Use first video
                features.video_width = meta.get('width')
                features.video_height = meta.get('height')
                features.frame_count = meta.get('frame_count')
                features.fps = meta.get('fps')
                
                if features.video_width and features.video_height and features.frame_count:
                    features.total_pixels = (features.video_width * features.video_height * 
                                            features.frame_count * features.num_videos)
        
        # Extract audio features
        if has_audios:
            audios = sample['audios']
            features.num_audios = len(audios) if isinstance(audios, list) else 1
            if 'audio_metadata' in sample:
                meta = sample['audio_metadata']
                if isinstance(meta, list) and meta:
                    meta = meta[0]
                features.audio_sample_rate = meta.get('sample_rate')
                features.audio_duration = meta.get('duration')
        
        # Estimate rough size in MB
        features.estimated_size_mb = self._estimate_size(features)
        
        return features
    
    def extract_from_batch(self, batch: Dict[str, Any]) -> SampleFeatures:
        """
        Extract features from a batched sample.
        
        Args:
            batch: Batched data dictionary where values are lists
            
        Returns:
            SampleFeatures object (aggregated)
        """
        # Determine batch size
        batch_size = 0
        for value in batch.values():
            if isinstance(value, list):
                batch_size = len(value)
                break
        
        if batch_size == 0:
            # Not a batched format, treat as single
            return self.extract_from_sample(batch)
        
        # Extract features from first sample and scale
        first_sample = {
            key: values[0] if isinstance(values, list) and values else values
            for key, values in batch.items()
        }
        
        features = self.extract_from_sample(first_sample)
        features.batch_size = batch_size
        
        # Scale certain features
        if features.estimated_size_mb:
            features.estimated_size_mb *= batch_size
        
        return features
    
    def _estimate_size(self, features: SampleFeatures) -> float:
        """
        Rough estimate of sample size in MB.
        
        This is a heuristic based on typical data sizes.
        """
        size_mb = 0.0
        
        # Text: ~1 byte per character
        if features.text_length:
            size_mb += features.text_length / (1024 * 1024)
        
        # Images: width * height * channels * bytes_per_pixel (typically 1-4)
        if features.total_pixels and features.modality in ['image', 'multimodal']:
            # Assume 3 bytes per pixel for RGB
            size_mb += (features.total_pixels * 3) / (1024 * 1024)
        
        # Videos: similar but multiplied by frames
        if features.total_pixels and features.modality == 'video':
            # Videos in memory are often decoded to raw frames
            size_mb += (features.total_pixels * 3) / (1024 * 1024)
        
        # Audio: sample_rate * duration * channels * bytes_per_sample
        if features.audio_sample_rate and features.audio_duration:
            # Assume 2 bytes per sample (16-bit), mono or stereo
            channels = 2
            bytes_per_sample = 2
            size_mb += (features.audio_sample_rate * features.audio_duration * 
                       channels * bytes_per_sample) / (1024 * 1024)
        
        return size_mb
    
    def analyze_batch_variance(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze variance in a batch to detect skew.
        
        High variance indicates need for dynamic batching.
        """
        if not any(isinstance(v, list) for v in batch.values()):
            return {'variance': 0, 'requires_dynamic_batching': False}
        
        # Extract features for each sample in batch
        batch_size = len(batch[next(iter(batch))])
        sizes = []
        
        for i in range(batch_size):
            sample = {k: (v[i] if isinstance(v, list) else v) for k, v in batch.items()}
            features = self.extract_from_sample(sample)
            if features.estimated_size_mb:
                sizes.append(features.estimated_size_mb)
        
        if not sizes:
            return {'variance': 0, 'requires_dynamic_batching': False}
        
        import numpy as np
        variance = float(np.var(sizes))
        mean_size = float(np.mean(sizes))
        coef_variation = variance / mean_size if mean_size > 0 else 0
        
        # High coefficient of variation suggests dynamic batching
        requires_dynamic = coef_variation > 0.5
        
        return {
            'variance': variance,
            'mean_size_mb': mean_size,
            'min_size_mb': float(np.min(sizes)),
            'max_size_mb': float(np.max(sizes)),
            'coef_variation': coef_variation,
            'requires_dynamic_batching': requires_dynamic,
        }
