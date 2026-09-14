"""Optional answer generation from hybrid retrieval results."""

from .ollama import OllamaConfig, generate_answer

__all__ = ["OllamaConfig", "generate_answer"]
