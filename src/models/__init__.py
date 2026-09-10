"""
Meta-Modeling & Validation Package.
Implements leak-free Purged Group Time-Series Cross-Validation and
L1-penalized Meta-Classification with out-of-sample probability calibration.
"""

from .cross_validation import PurgedGroupTimeSeriesSplit
from .meta_classifier import L1MetaClassifier, MetaModelValidator

__all__ = [
    "PurgedGroupTimeSeriesSplit",
    "L1MetaClassifier",
    "MetaModelValidator",
]