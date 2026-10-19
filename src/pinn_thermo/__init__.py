"""
PINNthermo: Input Convex Neural Network (ICNN) Helmholtz Free Energy & Peng-Robinson EOS Engine
"""

from .icnn_thermo import (
    InputConvexHelmholtzNetwork,
    ICNNState,
    ThermodynamicDerivatives
)
from .peng_robinson import (
    solve_peng_robinson_cubic,
    calculate_pr_fugacity_coefficient,
    PREOSResult
)
from .cape_open import CapeOpenPropertyPackage

__version__ = "1.0.0"
__all__ = [
    "InputConvexHelmholtzNetwork",
    "ICNNState",
    "ThermodynamicDerivatives",
    "solve_peng_robinson_cubic",
    "calculate_pr_fugacity_coefficient",
    "PREOSResult",
    "CapeOpenPropertyPackage",
    "__version__"
]
