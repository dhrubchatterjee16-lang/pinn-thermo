"""
PINNthermo CLI: Industrial Thermodynamics & ICNN Helmholtz Stability
"""

import argparse
import sys
import json
from .peng_robinson import solve_peng_robinson_cubic, calculate_pr_fugacity_coefficient
from .icnn_thermo import InputConvexHelmholtzNetwork, ICNNState
from .cape_open import CapeOpenPropertyPackage

def main():
    parser = argparse.ArgumentParser(
        description="PINNthermo: Thermodynamically Consistent Input Convex Neural Networks & Peng-Robinson EOS"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Command: eos
    p_eos = subparsers.add_parser("eos", help="Solve Peng-Robinson Cubic Equation of State")
    p_eos.add_argument("--temp", type=float, default=298.15, help="Temperature in Kelvin (default: 298.15)")
    p_eos.add_argument("--press", type=float, default=1.0e6, help="Pressure in Pascals (default: 1.0e6)")
    p_eos.add_argument("--tc", type=float, default=190.56, help="Critical temperature in K (default: 190.56 Methane)")
    p_eos.add_argument("--pc", type=float, default=45.99e5, help="Critical pressure in Pa (default: 45.99e5 Methane)")
    p_eos.add_argument("--omega", type=float, default=0.011, help="Acentric factor (default: 0.011)")

    # Command: icnn
    p_icnn = subparsers.add_parser("icnn", help="Evaluate ICNN Helmholtz Free Energy and Mechanical Stability")
    p_icnn.add_argument("--temp", type=float, default=300.0, help="Temperature in Kelvin")
    p_icnn.add_argument("--vol", type=float, default=0.024, help="Molar volume in m3/mol")
    p_icnn.add_argument("--moles", type=str, default="0.5,0.5", help="Comma-separated mole fractions (default: 0.5,0.5)")

    # Command: flash
    p_flash = subparsers.add_parser("flash", help="Perform CAPE-OPEN PT-Flash VLE separation")
    p_flash.add_argument("--temp", type=float, default=220.0, help="Temperature in Kelvin")
    p_flash.add_argument("--press", type=float, default=3.0e6, help="Pressure in Pascals")
    p_flash.add_argument("--compounds", type=str, default="Methane,Ethane", help="Comma-separated compounds")
    p_flash.add_argument("--feed", type=str, default="0.6,0.4", help="Comma-separated feed mole fractions")

    args = parser.parse_args()

    if args.command == "eos":
        res = solve_peng_robinson_cubic(t_k=args.temp, p_pa=args.press, tc_k=args.tc, pc_pa=args.pc, omega=args.omega)
        phi_vap = calculate_pr_fugacity_coefficient(args.temp, args.press, args.tc, args.pc, args.omega, res.z_vap)
        out = {
            "temperature_k": args.temp,
            "pressure_pa": args.press,
            "z_vapor": res.z_vap,
            "z_liquid": res.z_liq,
            "molar_volume_vap_m3_mol": res.molar_volume_vap,
            "molar_volume_liq_m3_mol": res.molar_volume_liq,
            "fugacity_coeff_vap": phi_vap,
            "has_phase_split": res.has_phase_split
        }
        print(json.dumps(out, indent=2))

    elif args.command == "icnn":
        moles = [float(x.strip()) for x in args.moles.split(",")]
        icnn = InputConvexHelmholtzNetwork(n_components=len(moles))
        state = ICNNState(temperature_k=args.temp, volume_m3=args.vol, mole_numbers=moles)
        derivs = icnn.evaluate(state)
        out = {
            "helmholtz_energy_j": derivs.helmholtz_energy_j,
            "pressure_pa": derivs.pressure_pa,
            "chemical_potentials_j_mol": derivs.chemical_potentials_j_mol,
            "mechanical_stability_hessian": derivs.mechanical_stability_hessian,
            "is_mechanically_stable": derivs.is_mechanically_stable,
            "gibbs_duhem_residual": derivs.gibbs_duhem_residual
        }
        print(json.dumps(out, indent=2))

    elif args.command == "flash":
        comps = [c.strip() for c in args.compounds.split(",")]
        feed = [float(x.strip()) for x in args.feed.split(",")]
        pkg = CapeOpenPropertyPackage(compounds=comps)
        res = pkg.calc_pt_flash(t_k=args.temp, p_pa=args.press, z_feed=feed)
        out = {
            "temperature_k": res.temperature_k,
            "pressure_pa": res.pressure_pa,
            "vapor_fraction": res.vapor_fraction,
            "liquid_fraction": res.liquid_fraction,
            "x_liquid": res.x_liquid,
            "y_vapor": res.y_vapor,
            "k_values": res.k_values,
            "converged": res.converged,
            "iterations": res.iterations
        }
        print(json.dumps(out, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
