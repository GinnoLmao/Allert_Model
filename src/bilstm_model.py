"""Compatibility re-export for the PyTorch Bi-LSTM model.

The original TensorFlow/Keras implementation has been replaced by
``src.pytorch_model``. Existing imports such as
``from src.bilstm_model import build_bilstm`` continue to work.
"""
from src.pytorch_model import (
    ALLERTBiLSTM as PyTorchBiLSTM,
    DEVICE,
    TemporalPatternAttention,
    build_bilstm,
    composite_loss,
    set_random_seeds,
)


__all__ = [
    "PyTorchBiLSTM",
    "DEVICE",
    "TemporalPatternAttention",
    "build_bilstm",
    "composite_loss",
    "set_random_seeds",
]
