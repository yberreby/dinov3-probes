import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class DINOv3LinearClassificationHead(nn.Linear, PyTorchModelHubMixin):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
