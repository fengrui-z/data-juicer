import importlib
import sys
import types
from unittest.mock import MagicMock

_HEAVY_MODULES = [
    "spacy",
    "sentencepiece",
    "transformers",
    "torch",
    "torchvision",
    "torchaudio",
    "ray",
    "ray.data",
    "ray.air",
    "nlpaug",
    "nlpcda",
    "kenlm",
    "toolz",
    "simhash",
    "pfld",
    "video_reader",
    "decord",
    "webrtcvad",
    "numba",
    "seaborn",
    "plotly",
    "dash",
    "gradio",
    "streamlit",
    "flagembedding",
    "sentence_transformers",
    "open_clip",
    "clip",
    "blip",
    "librosa",
    "soundfile",
    "resampy",
    "imageio",
    "cv2",
    "ffmpeg",
    "moviepy",
    "vit_pytorch",
    "taming",
    "diffusers",
    "accelerate",
    "einops",
    "timm",
    "basicsr",
    "facexlib",
    "realesrgan",
    "pdfplumber",
    "magic",
    "filetype",
    "cloudpickle",
    "redis",
    "boto3",
    "botocore",
    "google",
    "google.cloud",
    "marshmallow",
    "py7zr",
    "rarfile",
    "html2text",
    "markdown",
    "ruamel",
    "ruamel.yaml",
]

for mod_name in _HEAVY_MODULES:
    if mod_name not in sys.modules:
        mod = types.ModuleType(mod_name)
        mod.__path__ = []
        spec = importlib.machinery.ModuleSpec(mod_name, loader=None)
        spec.submodule_search_locations = []
        mod.__spec__ = spec
        sys.modules[mod_name] = mod

_ray = sys.modules["ray"]
_ray.remote = lambda *a, **kw: (lambda f: f)
_ray.init = lambda *a, **kw: None
_ray.shutdown = lambda *a, **kw: None
_ray.is_initialized = lambda: False
_ray.method = lambda *a, **kw: (lambda f: f)

class _FakeDataset:
    pass

class _FakeConfig:
    pass

_SUBMODULES = {
    "librosa.decompose": ["decompose"],
    "librosa.feature": ["melspectrogram", "chroma", "spectral"],
    "librosa.effects": ["trim", "split"],
    "ray.data": [],
    "ray.data._internal": [],
    "ray.data._internal.util": ["get_compute_strategy"],
    "ray.air": [],
    "torch.nn": ["Module"],
    "torch.optim": ["Adam"],
    "torch.utils": ["data"],
    "transformers.models": [],
    "transformers.models.auto": [],
    "transformers.models.auto.tokenization_auto": ["AutoTokenizer"],
    "transformers.models.auto.processing_auto": ["AutoProcessor"],
    "transformers.models.auto.modeling_auto": ["AutoModel"],
    "transformers.models.auto.image_processing_auto": ["AutoImageProcessor"],
}
for sub_name, attrs in _SUBMODULES.items():
    if sub_name not in sys.modules:
        mod = types.ModuleType(sub_name)
        mod.__path__ = []
        spec = importlib.machinery.ModuleSpec(sub_name, loader=None)
        spec.submodule_search_locations = []
        mod.__spec__ = spec
        for attr in attrs:
            setattr(mod, attr, MagicMock())
        sys.modules[sub_name] = mod

sys.modules["ray.data"].Dataset = _FakeDataset
sys.modules["ray.data"].read_api = MagicMock()
sys.modules["ray.air"].Config = _FakeConfig
