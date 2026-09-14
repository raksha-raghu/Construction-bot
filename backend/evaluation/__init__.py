"""Standalone retrieval evaluation; importing this package does not load models."""

from .dataset import load_dataset
from .runner import evaluate

__all__ = ["evaluate", "load_dataset"]
