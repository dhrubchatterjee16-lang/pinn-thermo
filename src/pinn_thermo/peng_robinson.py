"""
Peng-Robinson Equation of State (PR-EOS) & High-Pressure Vapor Fugacity
========================================================================
Solves the cubic equation of state:
  P = (R * T) / (v - b) - a(T) / (v * (v + b) + b * (v - b))

Z^3 - (1 - B)*Z^2 + (A - 3*B^2 - 2*B)*Z - (A*B - B^2 - B^3) = 0
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import numpy as np

R_GAS = 8.314462618 # J/(mol*K)

CRITICAL_DATA = {
    "Ethanol": {"Tc": 514.0, "Pc": 61.4e5, "omega": 0.644},
    "Water": {"Tc": 647.1, "Pc": 220.6e5, "omega": 0.344},
    "Methanol": {"Tc": 512.6, "Pc": 80.9e5, "omega": 0.565},
    "Acetone": {"Tc": 508.2, "Pc": 47.0e5, "omega": 0.307},
    "CO2": {"Tc": 304.1, "Pc": 73.8e5, "omega": 0.224},
    "Methane": {"Tc": 190.56, "Pc": 45.99e5, "omega": 0.011},
    "Ethane": {"Tc": 305.32, "Pc": 48.72e5, "omega": 0.099},
    "Propane": {"Tc": 369.83, "Pc": 42.48e5, "omega": 0.152}
}

@dataclass
class PREOSResult:
    component: str
    temperature_K: float
    pressure_bar: float
    reduced_temperature_Tr: float
    compressibility_factor_Z: float
    vapor_fugacity_coefficient_phi: float
    gas_phase_non_ideality_pct: float
    vle_regime: str

@dataclass
class PREOSRoots:
    z_vap: float
    z_liq: float
    molar_volume_vap: float
    molar_volume_liq: float
    has_phase_split: bool
    A: float
    B: float

def solve_peng_robinson_cubic(*args, **kwargs) -> Union[float, PREOSRoots]:
    """
    Solves PR cubic EOS. Supports two calling conventions:
    1. solve_peng_robinson_cubic(A, B) -> returns z_vap (float)
    2. solve_peng_robinson_cubic(t_k, p_pa, tc_k, pc_pa, omega) -> returns PREOSRoots
    """
    if len(args) == 2 and isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)) and not kwargs:
        A, B = float(args[0]), float(args[1])
        c2 = -(1.0 - B)
        c1 = A - 3.0 * (B ** 2) - 2.0 * B
        c0 = -(A * B - (B ** 2) - (B ** 3))
        roots = np.roots([1.0, c2, c1, c0])
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > B]
        return max(real_roots) if real_roots else 1.0

    # Physical parameters mode
    t_k = kwargs.get("t_k", args[0] if len(args) > 0 else 298.15)
    p_pa = kwargs.get("p_pa", args[1] if len(args) > 1 else 1.0e5)
    tc_k = kwargs.get("tc_k", args[2] if len(args) > 2 else 190.56)
    pc_pa = kwargs.get("pc_pa", args[3] if len(args) > 3 else 45.99e5)
    omega = kwargs.get("omega", args[4] if len(args) > 4 else 0.011)

    tr = t_k / tc_k
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * (omega ** 2)
    alpha = (1.0 + kappa * (1.0 - math.sqrt(max(1e-6, tr)))) ** 2

    a = 0.45724 * ((R_GAS * tc_k) ** 2) / pc_pa * alpha
    b = 0.07780 * (R_GAS * tc_k) / pc_pa

    A = a * p_pa / ((R_GAS * t_k) ** 2)
    B = b * p_pa / (R_GAS * t_k)

    c2 = -(1.0 - B)
    c1 = A - 3.0 * (B ** 2) - 2.0 * B
    c0 = -(A * B - (B ** 2) - (B ** 3))

    roots = np.roots([1.0, c2, c1, c0])
    real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-5 and r.real > B])

    if not real_roots:
        z_vap = z_liq = 1.0
    elif len(real_roots) == 1:
        z_vap = z_liq = real_roots[0]
    else:
        z_liq = real_roots[0]
        z_vap = real_roots[-1]

    mv_vap = (z_vap * R_GAS * t_k) / max(1.0, p_pa)
    mv_liq = (z_liq * R_GAS * t_k) / max(1.0, p_pa)

    return PREOSRoots(
        z_vap=round(float(z_vap), 4),
        z_liq=round(float(z_liq), 4),
        molar_volume_vap=round(float(mv_vap), 6),
        molar_volume_liq=round(float(mv_liq), 6),
        has_phase_split=bool(len(real_roots) > 1 and (z_vap - z_liq) > 1e-3),
        A=round(float(A), 6),
        B=round(float(B), 6)
    )

def calculate_pr_fugacity_coefficient(*args, **kwargs) -> Union[float, PREOSResult]:
    """
    Computes fugacity coefficient phi.
    Supports either:
    1. calculate_pr_fugacity_coefficient(component="Ethanol", temp_K=351.4, pres_bar=1.0) -> PREOSResult
    2. calculate_pr_fugacity_coefficient(t_k, p_pa, tc_k, pc_pa, omega, z_factor) -> float (phi)
    """
    if "component" in kwargs or (len(args) > 0 and isinstance(args[0], str)):
        comp = kwargs.get("component", args[0] if len(args) > 0 else "Ethanol")
        temp_k = kwargs.get("temp_K", args[1] if len(args) > 1 else 351.4)
        pres_bar = kwargs.get("pres_bar", args[2] if len(args) > 2 else 1.0)
        p_pa = pres_bar * 1e5

        props = CRITICAL_DATA.get(comp, CRITICAL_DATA["Ethanol"])
        tc_k = props["Tc"]
        pc_pa = props["Pc"]
        omega = props["omega"]

        roots = solve_peng_robinson_cubic(t_k=temp_k, p_pa=p_pa, tc_k=tc_k, pc_pa=pc_pa, omega=omega)
        z = roots.z_vap
        A, B = roots.A, roots.B

        sqrt2 = math.sqrt(2.0)
        term1 = z - 1.0 - math.log(max(1e-5, z - B))
        term2 = (A / (2.0 * sqrt2 * B)) * math.log(max(1e-5, (z + (1.0 + sqrt2) * B) / (z + (1.0 - sqrt2) * B)))
        ln_phi = term1 - term2
        phi = math.exp(np.clip(ln_phi, -20.0, 20.0))

        non_ideality = abs(1.0 - phi) * 100.0
        regime = "Ideal Gas Approximation Valid" if non_ideality < 3.0 else "High-Pressure Non-Ideal (PR Correction Mandatory)"

        return PREOSResult(
            component=comp,
            temperature_K=temp_k,
            pressure_bar=pres_bar,
            reduced_temperature_Tr=round(temp_k / tc_k, 3),
            compressibility_factor_Z=round(z, 4),
            vapor_fugacity_coefficient_phi=round(phi, 4),
            gas_phase_non_ideality_pct=round(non_ideality, 2),
            vle_regime=regime
        )

    # Physical signature
    t_k = kwargs.get("t_k", args[0] if len(args) > 0 else 298.15)
    p_pa = kwargs.get("p_pa", args[1] if len(args) > 1 else 1.0e5)
    tc_k = kwargs.get("tc_k", args[2] if len(args) > 2 else 190.56)
    pc_pa = kwargs.get("pc_pa", args[3] if len(args) > 3 else 45.99e5)
    omega = kwargs.get("omega", args[4] if len(args) > 4 else 0.011)
    z = kwargs.get("z_factor", args[5] if len(args) > 5 else 1.0)

    tr = t_k / tc_k
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * (omega ** 2)
    alpha = (1.0 + kappa * (1.0 - math.sqrt(max(1e-6, tr)))) ** 2

    a = 0.45724 * ((R_GAS * tc_k) ** 2) / pc_pa * alpha
    b = 0.07780 * (R_GAS * tc_k) / pc_pa

    A = a * p_pa / ((R_GAS * t_k) ** 2)
    B = b * p_pa / (R_GAS * t_k)

    sqrt2 = math.sqrt(2.0)
    term1 = z - 1.0 - math.log(max(1e-5, z - B))
    denom = max(1e-5, z + (1.0 - sqrt2) * B)
    num = max(1e-5, z + (1.0 + sqrt2) * B)
    term2 = (A / (2.0 * sqrt2 * max(1e-8, B))) * math.log(num / denom)
    ln_phi = term1 - term2
    return float(math.exp(np.clip(ln_phi, -20.0, 20.0)))
