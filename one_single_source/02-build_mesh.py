#!/usr/bin/env python3
"""
02_build_mesh.py

Build a 2D layered SEM mesh using Salvus and write it to an HDF5 mesh file.

Date: 10 Mar 2026
Author: Shi Yao (yaoshi229@gmail.com)

Inputs
- config.yaml

Outputs
- mesh.mesh_file: Salvus mesh in HDF5 format

Notes
- Material properties:
  vp is set in config.yaml; vs is derived from vp/vp_vs_ratio;
  density is computed from Brocher (2005) rho(vp) polynomial if enabled.
"""

from pathlib import Path

import yaml
import salvus.namespace as sn


def brocher_rho_from_vp(vp_m_s: float) -> float:
    """
    Brocher (2005) polynomial: rho(vp).
    Input:  vp in m/s
    Output: rho in kg/m^3
    """
    vp = vp_m_s / 1000.0  # km/s
    rho_g_cm3 = (
        1.6612 * vp
        - 0.4721 * vp**2
        + 0.0671 * vp**3
        - 0.0043 * vp**4
        + 0.000106 * vp**5
    )
    return rho_g_cm3 * 1000.0  # kg/m^3


def main():
    # ---------------------------
    # Load config.yaml (local)
    # ---------------------------
    cfg_path = Path(__file__).with_name("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find {cfg_path}. Put config.yaml in the same directory as this script.")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    # ---------------------------
    # Domain from config
    # ---------------------------
    dom = cfg["domain"]
    d = sn.domain.dim2.BoxDomain(
        x0=float(dom["x0"]),
        x1=float(dom["x1"]),
        y0=float(dom["y0"]),
        y1=float(dom["y1"]),
    )

    ymin, ymax = float(dom["y0"]), float(dom["y1"])

    # ---------------------------
    # Material from config
    # ---------------------------
    vp = float(cfg["materials"]["crust"]["vp_m_s"])
    vs = vp / float(cfg["materials"]["crust"]["vp_vs_ratio"])

    if bool(cfg["materials"]["crust"]["rho_from_brocher"]):
        rho = brocher_rho_from_vp(vp)
    else:
        rho = float(cfg["materials"]["crust"]["rho_kg_m3"])

    crust = sn.material.from_params(vp=vp, vs=vs, rho=rho)

    # Layered model (single layer)
    top = sn.layered_meshing.interface.Hyperplane.at(float(ymax))
    bottom = sn.layered_meshing.interface.Hyperplane.at(float(ymin))
    layered_model = sn.layered_meshing.LayeredModel(strata=[top, crust, bottom])

    # ---------------------------
    # Meshing parameters
    # ---------------------------
    mesh_cfg = cfg["mesh"]
    mesh = sn.layered_meshing.mesh_from_domain(
        domain=d,
        model=sn.layered_meshing.MeshingProtocol(layered_model),
        mesh_resolution=sn.MeshResolution(
            reference_frequency=float(mesh_cfg["reference_frequency_hz"]),
            elements_per_wavelength=float(mesh_cfg["elements_per_wavelength"]),
        ),
    )

    # ---------------------------
    # Write mesh
    # ---------------------------
    mesh_file = Path(mesh_cfg["mesh_file"])
    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_h5(str(mesh_file))

    print(f"Wrote mesh: {mesh_file}")
    print(f"Material: vp={vp:.1f} m/s, vs={vs:.1f} m/s, rho={rho:.1f} kg/m^3")


if __name__ == "__main__":
    main()
