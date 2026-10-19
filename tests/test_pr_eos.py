"""
Tests for Peng-Robinson EOS & CAPE-OPEN PT-Flash Engine
"""

import pytest
import math
from pinn_thermo.peng_robinson import solve_peng_robinson_cubic, calculate_pr_fugacity_coefficient
from pinn_thermo.cape_open import CapeOpenPropertyPackage

def test_pr_ideal_gas_limit():
    """
    At very high temperature (T >> Tc) and low pressure (P << Pc), Z should approach 1.0.
    """
    tc = 190.56  # Methane
    pc = 45.99e5
    omega = 0.011

    res = solve_peng_robinson_cubic(t_k=1000.0, p_pa=1000.0, tc_k=tc, pc_pa=pc, omega=omega)
    assert abs(res.z_vap - 1.0) < 0.01
    phi = calculate_pr_fugacity_coefficient(t_k=1000.0, p_pa=1000.0, tc_k=tc, pc_pa=pc, omega=omega, z_factor=res.z_vap)
    assert abs(phi - 1.0) < 0.01

def test_pr_two_phase_split():
    """
    Under subcritical conditions (T < Tc, moderate P), cubic should exhibit distinct liquid and vapor roots.
    """
    tc = 369.83  # Propane
    pc = 42.48e5
    omega = 0.152

    res = solve_peng_robinson_cubic(t_k=300.0, p_pa=1.0e6, tc_k=tc, pc_pa=pc, omega=omega)
    assert res.z_vap > res.z_liq
    assert res.molar_volume_vap > res.molar_volume_liq
    assert res.has_phase_split

def test_cape_open_pt_flash():
    """
    Verify isothermal-isobaric flash for Methane/Ethane binary mixture.
    """
    pkg = CapeOpenPropertyPackage(compounds=["Methane", "Ethane"])
    flash = pkg.calc_pt_flash(t_k=220.0, p_pa=3.0e6, z_feed=[0.5, 0.5])

    assert flash.converged
    assert 0.0 <= flash.vapor_fraction <= 1.0
    assert abs(flash.vapor_fraction + flash.liquid_fraction - 1.0) < 1e-4
    assert abs(sum(flash.x_liquid) - 1.0) < 1e-4
    assert abs(sum(flash.y_vapor) - 1.0) < 1e-4

def test_cape_open_single_phase():
    """
    Verify single-phase property calculation returns physical consistency.
    """
    pkg = CapeOpenPropertyPackage(compounds=["Methane", "CarbonDioxide"])
    props = pkg.calc_single_phase_properties(t_k=320.0, p_pa=5.0e6, z=[0.8, 0.2])

    assert props["z_vapor"] > 0.0
    assert props["icnn_is_mechanically_stable"]
    assert props["icnn_stability_hessian"] >= 0.0
