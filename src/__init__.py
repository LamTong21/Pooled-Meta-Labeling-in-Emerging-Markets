"""
Core quantitative framework for Pooled Cross-Sectional Meta-Labeling
and Microstructure Scaffolding in Emerging Equity Markets.
"""

from .data_engine import MultiAssetDataPreparer
from .labeling import TripleBarrierLabeler

__all__ = [
    "MultiAssetDataPreparer",
    "TripleBarrierLabeler",
]

__version__ = "1.0.0"