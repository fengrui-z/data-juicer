import pytest

from data_juicer.core.elasticjuicer.predictor.feature_extractor import (
    FeatureExtractor,
    SampleFeatures,
)


@pytest.fixture
def extractor():
    return FeatureExtractor()


def test_text_sample(extractor):
    sample = {"text": "hello world this is a test"}
    features = extractor.extract_from_sample(sample)
    assert features.modality == "text"
    assert features.text_length == len("hello world this is a test")
    assert features.num_tokens == 6
    assert features.estimated_size_mb is not None
    assert features.estimated_size_mb > 0


def test_image_sample_with_metadata(extractor):
    sample = {
        "text": "",
        "images": ["img1.jpg"],
        "image_metadata": [{"width": 640, "height": 480, "channels": 3}],
    }
    features = extractor.extract_from_sample(sample)
    assert features.modality == "image"
    assert features.num_images == 1
    assert features.image_width == 640
    assert features.image_height == 480
    assert features.total_pixels == 640 * 480 * 1


def test_video_sample(extractor):
    sample = {
        "videos": ["vid1.mp4"],
        "video_metadata": [{"width": 1920, "height": 1080, "frame_count": 300, "fps": 30.0}],
    }
    features = extractor.extract_from_sample(sample)
    assert features.modality == "video"
    assert features.num_videos == 1
    assert features.frame_count == 300
    assert features.fps == 30.0
    assert features.total_pixels == 1920 * 1080 * 300 * 1


def test_audio_sample(extractor):
    sample = {
        "audios": ["audio1.wav"],
        "audio_metadata": [{"sample_rate": 44100, "duration": 5.0}],
    }
    features = extractor.extract_from_sample(sample)
    assert features.modality == "audio"
    assert features.audio_sample_rate == 44100
    assert features.audio_duration == 5.0


def test_multimodal_sample(extractor):
    sample = {
        "text": "a photo of a cat",
        "images": ["cat.jpg"],
    }
    features = extractor.extract_from_sample(sample)
    assert features.modality == "multimodal"


def test_empty_metadata(extractor):
    sample = {
        "images": ["img.jpg"],
    }
    features = extractor.extract_from_sample(sample)
    assert features.modality == "image"
    assert features.num_images == 1
    assert features.image_width is None


def test_empty_sample(extractor):
    sample = {}
    features = extractor.extract_from_sample(sample)
    assert features.text_length is None
    assert features.num_images == 0


def test_feature_vector_length(extractor):
    sample = {"text": "test"}
    features = extractor.extract_from_sample(sample)
    vec = features.to_feature_vector()
    assert len(vec) == len(SampleFeatures.feature_names())
    assert len(vec) == 17


def test_batch_extraction(extractor):
    batch = {
        "text": ["hello", "world", "test"],
        "images": [["a.jpg"], ["b.jpg"], ["c.jpg"]],
    }
    features = extractor.extract_from_batch(batch)
    assert features.batch_size == 3


def test_analyze_batch_variance(extractor):
    batch = {
        "text": ["short", "a" * 10000],
    }
    result = extractor.analyze_batch_variance(batch)
    assert "variance" in result
    assert "requires_dynamic_batching" in result


def test_sample_features_to_feature_vector_default():
    features = SampleFeatures()
    vec = features.to_feature_vector()
    assert vec[0] == 1.0
    assert all(v == 0.0 for v in vec[1:])
