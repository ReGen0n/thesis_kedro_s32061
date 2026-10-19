"""data_science pipeline: split -> train regressor -> evaluate -> permutation importance."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
