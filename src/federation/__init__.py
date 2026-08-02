"""
CAMFS M1 — Federation Package Exporter
======================================

Exports federation protocol modules:
  - FederatedPhaseState, PhaseController
  - FederatedClient
  - FederatedServer
"""

from src.federation.client import FederatedClient
from src.federation.phase_controller import FederatedPhaseState, PhaseController
from src.federation.server import FederatedServer

__all__ = [
    "FederatedPhaseState",
    "PhaseController",
    "FederatedClient",
    "FederatedServer",
]
