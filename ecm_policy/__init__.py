"""TwinTeleop ECM 2027 policy package (v4: official gRPC client adapter).

The gRPC client lives in ``ecm_policy.grpc_client`` and is intentionally
NOT imported here: it needs protobuf, which is not installed on every
host that runs this package's tests (the data-sample venv has h5py but
no protobuf, and PyPI is unreachable from this host to install it).
Import it directly: ``from ecm_policy.grpc_client import ChamberGrpcPolicy``.
"""

from .core import ReplayFollowCore
from .multidemo import MultiDemoCore

__all__ = ["ReplayFollowCore", "MultiDemoCore"]
__version__ = "0.4.0"
