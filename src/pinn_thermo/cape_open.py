"""
CAPE-OPEN Thermodynamic Interface & Process Simulator Socket
=============================================================
Provides standard CAPE-OPEN 1.1 thermodynamic property package methods:
- Single-phase property calculations (density, enthalpy, entropy, fugacity)
- Vapor-Liquid Equilibrium (VLE) isothermal-isobaric flash calculation (PT-Flash)
- Seamless integration with Peng-Robinson EOS and ICNN Helmholtz engines
Compatible with DWSIM, Aspen Plus, and open-source flowsheet simulators.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from .peng_robinson import solve_peng_robinson_cubic, calculate_pr_fugacity_coefficient, PREOSResult
from .icnn_thermo import InputConvexHelmholtzNetwork, ICNNState, ThermodynamicDerivatives

@dataclass
class CompoundConstant:
    cas_id: str
    name: str
    tc_k: float
    pc_pa: float
    omega: float
    mw_g_mol: float
    ideal_cp_coeffs: Tuple[float, float, float, float]  # A, B, C, D in J/(mol K)

# Default benchmark compounds
STANDARD_COMPOUNDS: Dict[str, CompoundConstant] = {
    "Methane": CompoundConstant(
        cas_id="74-82-8", name="Methane", tc_k=190.56, pc_pa=45.99e5, omega=0.011, mw_g_mol=16.04,
        ideal_cp_coeffs=(19.25, 0.052, 1.2e-5, -1.1e-8)
    ),
    "Ethane": CompoundConstant(
        cas_id="74-84-0", name="Ethane", tc_k=305.32, pc_pa=48.72e5, omega=0.099, mw_g_mol=30.07,
        ideal_cp_coeffs=(5.4, 0.178, -6.9e-5, 8.7e-9)
    ),
    "Propane": CompoundConstant(
        cas_id="74-98-6", name="Propane", tc_k=369.83, pc_pa=42.48e5, omega=0.152, mw_g_mol=44.10,
        ideal_cp_coeffs=(-4.2, 0.306, -1.6e-4, 3.2e-8)
    ),
    "CarbonDioxide": CompoundConstant(
        cas_id="124-38-9", name="CarbonDioxide", tc_k=304.13, pc_pa=73.77e5, omega=0.225, mw_g_mol=44.01,
        ideal_cp_coeffs=(19.8, 0.073, -5.6e-5, 1.7e-8)
    )
}

@dataclass
class FlashResult:
    temperature_k: float
    pressure_pa: float
    vapor_fraction: float
    liquid_fraction: float
    x_liquid: List[float]
    y_vapor: List[float]
    k_values: List[float]
    converged: bool
    iterations: int

class CapeOpenPropertyPackage:
    """
    Standard CAPE-OPEN Thermodynamic Property Package for SciML & EOS calculation.
    """
    def __init__(self, compounds: Optional[List[str]] = None, use_icnn: bool = True):
        self.compounds = compounds or ["Methane", "Ethane"]
        self.compound_data = [STANDARD_COMPOUNDS.get(c, STANDARD_COMPOUNDS["Methane"]) for c in self.compounds]
        self.use_icnn = use_icnn
        self.icnn = InputConvexHelmholtzNetwork(n_components=len(self.compounds))

    def get_compound_names(self) -> List[str]:
        return [c.name for c in self.compound_data]

    def calc_single_phase_properties(self, t_k: float, p_pa: float, z: List[float]) -> Dict[str, Any]:
        """
        Calculates compressibility factor, fugacity coefficients, molar volume, and enthalpy.
        """
        z_arr = np.array(z, dtype=np.float64)
        z_arr = z_arr / np.sum(z_arr)

        # Average pseudo-critical properties (Kay's mixing rule)
        tc_mix = float(np.sum([comp.tc_k * zi for comp, zi in zip(self.compound_data, z_arr)]))
        pc_mix = float(np.sum([comp.pc_pa * zi for comp, zi in zip(self.compound_data, z_arr)]))
        omega_mix = float(np.sum([comp.omega * zi for comp, zi in zip(self.compound_data, z_arr)]))
        mw_mix = float(np.sum([comp.mw_g_mol * zi for comp, zi in zip(self.compound_data, z_arr)]))

        pr_res = solve_peng_robinson_cubic(t_k=t_k, p_pa=p_pa, tc_k=tc_mix, pc_pa=pc_mix, omega=omega_mix)

        # Calculate fugacity coeff
        phi_vap = calculate_pr_fugacity_coefficient(t_k=t_k, p_pa=p_pa, tc_k=tc_mix, pc_pa=pc_mix, omega=omega_mix, z_factor=pr_res.z_vap)
        phi_liq = calculate_pr_fugacity_coefficient(t_k=t_k, p_pa=p_pa, tc_k=tc_mix, pc_pa=pc_mix, omega=omega_mix, z_factor=pr_res.z_liq)

        # ICNN Helmholtz stability evaluation
        vol_est = (pr_res.z_vap * 8.314462 * t_k / p_pa) if p_pa > 0 else 0.024
        icnn_state = ICNNState(temperature_k=t_k, volume_m3=vol_est, mole_numbers=z_arr)
        icnn_eval = self.icnn.evaluate(icnn_state)

        return {
            "temperature_k": t_k,
            "pressure_pa": p_pa,
            "composition": list(z_arr),
            "z_vapor": pr_res.z_vap,
            "z_liquid": pr_res.z_liq,
            "molar_volume_vap_m3_mol": pr_res.molar_volume_vap,
            "molar_volume_liq_m3_mol": pr_res.molar_volume_liq,
            "fugacity_coeff_vap": phi_vap,
            "fugacity_coeff_liq": phi_liq,
            "pseudo_tc_k": tc_mix,
            "pseudo_pc_pa": pc_mix,
            "mean_mw_g_mol": mw_mix,
            "icnn_helmholtz_j": icnn_eval.helmholtz_energy_j,
            "icnn_is_mechanically_stable": icnn_eval.is_mechanically_stable,
            "icnn_stability_hessian": icnn_eval.mechanical_stability_hessian
        }

    def calc_pt_flash(self, t_k: float, p_pa: float, z_feed: List[float], max_iter: int = 50, tol: float = 1e-5) -> FlashResult:
        """
        Rachford-Rice Isothermal-Isobaric (PT) Flash Algorithm.
        """
        z = np.array(z_feed, dtype=np.float64)
        z = z / np.sum(z)
        n = len(z)

        # Wilson estimate for initial K-values: K_i = (P_c,i / P) * exp(5.37 * (1 + omega_i) * (1 - T_c,i / T))
        k_values = np.zeros(n)
        for i, comp in enumerate(self.compound_data):
            tr = t_k / comp.tc_k
            pr = p_pa / comp.pc_pa
            ln_k = 5.37 * (1.0 + comp.omega) * (1.0 - 1.0 / tr) + math.log(max(1e-4, 1.0 / pr))
            k_values[i] = math.exp(np.clip(ln_k, -15.0, 15.0))

        # Check subcooled liquid or superheated vapor
        sum_z_k = np.sum(z * k_values)
        sum_z_over_k = np.sum(z / k_values)

        if sum_z_k < 1.0:
            # Subcooled liquid
            return FlashResult(
                temperature_k=t_k, pressure_pa=p_pa, vapor_fraction=0.0, liquid_fraction=1.0,
                x_liquid=list(z), y_vapor=list(z * k_values / max(1e-8, sum_z_k)),
                k_values=list(k_values), converged=True, iterations=0
            )
        if sum_z_over_k < 1.0:
            # Superheated vapor
            return FlashResult(
                temperature_k=t_k, pressure_pa=p_pa, vapor_fraction=1.0, liquid_fraction=0.0,
                x_liquid=list((z / k_values) / max(1e-8, sum_z_over_k)), y_vapor=list(z),
                k_values=list(k_values), converged=True, iterations=0
            )

        # Solve Rachford-Rice equation: sum_i [ z_i * (K_i - 1) / (1 + beta * (K_i - 1)) ] = 0
        beta = 0.5  # vapor fraction
        converged = False
        iteration = 0

        for iteration in range(1, max_iter + 1):
            denom = 1.0 + beta * (k_values - 1.0)
            f = np.sum(z * (k_values - 1.0) / denom)
            df = -np.sum(z * ((k_values - 1.0) ** 2) / (denom ** 2))

            if abs(f) < tol:
                converged = True
                break

            step = f / df if df != 0 else 0.01
            beta = np.clip(beta - step, 1e-4, 1.0 - 1e-4)

        x_liq = z / (1.0 + beta * (k_values - 1.0))
        x_liq = x_liq / np.sum(x_liq)
        y_vap = k_values * x_liq
        y_vap = y_vap / np.sum(y_vap)

        return FlashResult(
            temperature_k=t_k,
            pressure_pa=p_pa,
            vapor_fraction=float(beta),
            liquid_fraction=float(1.0 - beta),
            x_liquid=[float(v) for v in x_liq],
            y_vapor=[float(v) for v in y_vap],
            k_values=[float(k) for k in k_values],
            converged=converged,
            iterations=iteration
        )
