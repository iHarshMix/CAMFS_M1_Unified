"""
CAMFS M1 — Models Package Exporter
===================================

Exports core neural network modules:
  - UnimodalEncoder, ConvBlock
  - PrototypeBank
  - SubsetFusionHead
  - UNetDecoder
"""

from src.models.decoder import UNetDecoder
from src.models.encoder import ConvBlock, UnimodalEncoder
from src.models.fusion import CANONICAL_MODALITY_ORDER, SubsetFusionHead
from src.models.prototypes import PrototypeBank, normalize_l2

__all__ = [
    "ConvBlock",
    "UnimodalEncoder",
    "PrototypeBank",
    "normalize_l2",
    "SubsetFusionHead",
    "CANONICAL_MODALITY_ORDER",
    "UNetDecoder",
]
