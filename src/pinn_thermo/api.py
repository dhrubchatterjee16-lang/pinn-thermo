"""
FastAPI REST Service for PINNthermo
"""

from typing import List, Optional
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
except ImportError:
    FastAPI = None

from .peng_robinson import solve_peng_robinson_cubic, calculate_pr_fugacity_coefficient
from .icnn_thermo import InputConvexHelmholtzNetwork, ICNNState
from .cape_open import CapeOpenPropertyPackage

if FastAPI is not None:
    app = FastAPI(
        title="PINNthermo API",
        description="Thermodynamically Consistent Input Convex Neural Networks & CAPE-OPEN Flash Engine",
        version="1.0.0"
    )

    class EOSRequest(BaseModel):
        temperature_k: float = Field(298.15, description="Temperature in K")
        pressure_pa: float = Field(1.0e6, description="Pressure in Pa")
        tc_k: float = Field(190.56, description="Critical Temperature in K")
        pc_pa: float = Field(45.99e5, description="Critical Pressure in Pa")
        omega: float = Field(0.011, description="Acentric factor")

    class ICNNRequest(BaseModel):
        temperature_k: float = Field(300.0, description="Temperature in K")
        volume_m3: float = Field(0.024, description="Molar volume in m3")
        mole_numbers: List[float] = Field([0.5, 0.5], description="Component molar fractions")

    class FlashRequest(BaseModel):
        temperature_k: float = Field(220.0, description="Flash temperature in K")
        pressure_pa: float = Field(3.0e6, description="Flash pressure in Pa")
        compounds: List[str] = Field(["Methane", "Ethane"], description="Mixture components")
        feed: List[float] = Field([0.6, 0.4], description="Feed composition fractions")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "pinn-thermo", "version": "1.0.0"}

    @app.post("/predict/eos")
    def predict_eos(req: EOSRequest):
        res = solve_peng_robinson_cubic(req.temperature_k, req.pressure_pa, req.tc_k, req.pc_pa, req.omega)
        phi_vap = calculate_pr_fugacity_coefficient(req.temperature_k, req.pressure_pa, req.tc_k, req.pc_pa, req.omega, res.z_vap)
        return {
            "z_vapor": res.z_vap,
            "z_liquid": res.z_liq,
            "molar_volume_vap_m3_mol": res.molar_volume_vap,
            "molar_volume_liq_m3_mol": res.molar_volume_liq,
            "fugacity_coeff_vap": phi_vap,
            "has_phase_split": res.has_phase_split
        }

    @app.post("/predict/icnn")
    def predict_icnn(req: ICNNRequest):
        icnn = InputConvexHelmholtzNetwork(n_components=len(req.mole_numbers))
        state = ICNNState(temperature_k=req.temperature_k, volume_m3=req.volume_m3, mole_numbers=req.mole_numbers)
        derivs = icnn.evaluate(state)
        return {
            "helmholtz_energy_j": derivs.helmholtz_energy_j,
            "pressure_pa": derivs.pressure_pa,
            "chemical_potentials_j_mol": derivs.chemical_potentials_j_mol,
            "mechanical_stability_hessian": derivs.mechanical_stability_hessian,
            "is_mechanically_stable": derivs.is_mechanically_stable,
            "gibbs_duhem_residual": derivs.gibbs_duhem_residual
        }

    @app.post("/predict/flash")
    def predict_flash(req: FlashRequest):
        pkg = CapeOpenPropertyPackage(compounds=req.compounds)
        res = pkg.calc_pt_flash(req.temperature_k, req.pressure_pa, req.feed)
        return {
            "vapor_fraction": res.vapor_fraction,
            "liquid_fraction": res.liquid_fraction,
            "x_liquid": res.x_liquid,
            "y_vapor": res.y_vapor,
            "k_values": res.k_values,
            "converged": res.converged,
            "iterations": res.iterations
        }
else:
    app = None
