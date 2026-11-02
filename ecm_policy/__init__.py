"""TwinTeleop ECM 2027 policy package (v3: multi-demo core wired into both adapters)."""

from .core import ReplayFollowCore
from .multidemo import MultiDemoCore

__all__ = ["ReplayFollowCore", "MultiDemoCore"]
__version__ = "0.3.0"
