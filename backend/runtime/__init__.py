"""Runtime package exports."""

from .batch_generator import NPCBatchGenerator, get_batch_generator
from .state_manager import NPCStateManager, get_state_manager

__all__ = [
    "NPCBatchGenerator",
    "NPCStateManager",
    "get_batch_generator",
    "get_state_manager",
]
