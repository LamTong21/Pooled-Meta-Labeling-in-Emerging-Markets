"""
Feature Engineering and Econometric Scaffolding Package.
Decouples isolated technical indicators and market microstructure features 
from strictly leak-free in-fold statistical variable selection.
"""

from .technical import generate_primary_signals, compute_technical_indicators
from .microstructure import extract_microstructure_features
from .econometric_pipeline import EconometricsFeaturePipeline

__all__ = [
    "generate_primary_signals",
    "compute_technical_indicators",
    "extract_microstructure_features",
    "EconometricsFeaturePipeline",
]