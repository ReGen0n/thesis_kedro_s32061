"""data_processing pipeline: raw patients CSV -> validated -> model input table."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
