"""
Tests for Input Convex Neural Network (ICNN) Thermodynamic Convexity
"""

import pytest
import numpy as np
from pinn_thermo.icnn_thermo import InputConvexHelmholtzNetwork, ICNNState

def test_icnn_mechanical_stability_unconditional():
    """
    Verify that d^2 A / dV^2 >= 0 holds unconditionally across multiple temperatures and volumes.
    """
    icnn = InputConvexHelmholtzNetwork(n_components=2, seed=42)

    temperatures = [150.0, 250.0, 300.0, 450.0, 600.0]
    volumes = [0.001, 0.005, 0.024, 0.1, 0.5]
    moles = np.array([0.7, 0.3])

    for T in temperatures:
        for V in volumes:
            state = ICNNState(temperature_k=T, volume_m3=V, mole_numbers=moles)
            derivs = icnn.evaluate(state)
            assert derivs.is_mechanically_stable, f"Failed mechanical stability at T={T}, V={V}"
            assert derivs.mechanical_stability_hessian >= 0.0

def test_icnn_derivatives_physical_bounds():
    """
    Verify Helmholtz energy and chemical potentials return finite physical numbers.
    """
    icnn = InputConvexHelmholtzNetwork(n_components=3, seed=123)
    state = ICNNState(temperature_k=298.15, volume_m3=0.024, mole_numbers=np.array([0.5, 0.3, 0.2]))
    derivs = icnn.evaluate(state)

    assert not np.isnan(derivs.helmholtz_energy_j)
    assert not np.isinf(derivs.helmholtz_energy_j)
    assert len(derivs.chemical_potentials_j_mol) == 3
    for mu in derivs.chemical_potentials_j_mol:
        assert not np.isnan(mu)

def test_icnn_volume_expansion_pressure():
    """
    Verify that increasing molar volume generally leads to lower or monotonic pressure relaxation.
    """
    icnn = InputConvexHelmholtzNetwork(n_components=1, seed=99)
    moles = np.array([1.0])
    v_dense = ICNNState(temperature_k=300.0, volume_m3=0.002, mole_numbers=moles)
    v_dilute = ICNNState(temperature_k=300.0, volume_m3=0.050, mole_numbers=moles)

    res_dense = icnn.evaluate(v_dense)
    res_dilute = icnn.evaluate(v_dilute)

    # d^2A/dV^2 >= 0 implies -dP/dV >= 0, so dP/dV <= 0 (pressure decreases with volume)
    assert res_dense.pressure_pa >= res_dilute.pressure_pa
