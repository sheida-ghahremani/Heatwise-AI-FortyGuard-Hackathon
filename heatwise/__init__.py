"""HeatWise AI: UTCI-based outdoor thermal-stress-aware routing."""

from .models import Activity, AgeGroup, RoutePreference, UserProfile
from .risk import segment_heat_cost

__all__ = [
    "Activity",
    "AgeGroup",
    "RoutePreference",
    "UserProfile",
    "segment_heat_cost",
]
