# -*- coding: utf-8 -*-
"""
06-eval_NN.py

Reads the Newtonian noise files and analyzes them. Produces plots.

Inputs
- config.yaml 
- output_dir/results/[project_name]_[NN_tag]/[NN_data].npy

Outputs
- fig_dir/NN_results[project_name]_[NN_tag]/[NN_plots].svg

Date: 11 Mar 2026
Author: Patrick Schillings (patrick.schillings@rwth-aachen.de)
"""

import yaml
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.constants as co
import scipy.signal as sig
import time as systime


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
    
    starttime = systime.time()
    plt.close("all")
    
    # === Plotting defaults ===
    plt.rc('legend',fontsize=22)
    plt.rc('axes',labelsize=25,titlesize=25)
    plt.rc("xtick",labelsize=20)
    plt.rc("ytick",labelsize=20)
    plt.rc('figure',figsize=(10,9))
    plt.rc('font',size=30)
    
    
    # ---------------------------
    # Load config.yaml (local)
    # ---------------------------
    
    cfg_path = Path(__file__).with_name("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find {cfg_path}. Put config.yaml in the same directory as this script.")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)


    # ---------------------------
    # Load and define paths
    # ---------------------------
    dataset = cfg["project"]["name"]
    tag = dataset + str(cfg["Newtonian_noise"]["tag"])
    
    out_dir = Path(cfg["paths"]["output_dir"])
    derived_dir = out_dir / "derived"    
    dis_path = derived_dir / "displacement.h5"
    result_dir = out_dir / "results"
    tag_dir = result_dir / tag
    
    fig_dir = Path(cfg["paths"]["fig_dir"]) / "NN_results" / tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    
    # ---------------------------
    # Load and define parameters
    # ---------------------------

    # === Constants ===
    pi = np.pi
    G = co.gravitational_constant
    M = cfg["Newtonian_noise"]["test_mass"]["M"]
    vp = cfg["materials"]["crust"]["vp_m_s"]
    
    if bool(cfg["materials"]["crust"]["rho_from_brocher"]):
        rho = brocher_rho_from_vp(vp)
    else:
        rho = float(cfg["materials"]["crust"]["rho_kg_m3"])
        
        
    scaling = float(cfg["Newtonian_noise"]["scaling"])
        
    # === Test mass geometry ===
    mirror_positions = np.array(cfg["Newtonian_noise"]["test_mass"]["location"])
    mirror_directions = np.array(cfg["Newtonian_noise"]["test_mass"]["orientation"])
    mirror_count = len(mirror_positions)
    
    if len(mirror_positions[0]) == 2:
        mirror_positions = np.concatenate((mirror_positions.T,np.zeros((1,mirror_count)))).T
    if len(mirror_directions[0]) == 2:
        mirror_directions = np.concatenate((mirror_directions.T,np.zeros((1,mirror_count)))).T
    
    
    # ---------------------------
    # Load data
    # ---------------------------
    
    # === Load forces and local displacement ===
    forces_bulk = scaling * np.load(tag_dir/"bulk_forces.npy")
    forces_dipole = scaling * np.load(tag_dir/"dipole_forces.npy")
    
    loc_x_disp = scaling * np.load(tag_dir/"x_displacements.npy")
    loc_y_disp = scaling * np.load(tag_dir/"y_displacements.npy")
    loc_div = scaling * np.load(tag_dir/"divergences.npy")
    loc_curl = scaling * np.load(tag_dir/"curls.npy")
    
    loc_disp = np.einsum("ij,jil->il", mirror_directions, np.array([loc_x_disp, loc_y_disp, np.zeros_like(loc_x_disp)]))
    
    # === Load wavefields and time ===
    wf_ds = xr.load_dataset(dis_path, engine="h5netcdf")
    wf = wf_ds["__xarray_dataarray_variable__"]
    
    time = np.array(wf["t"])
    
    Nt = len(time)
    dt = (time[-1] - time[0])/(Nt-1)
    
    tmin = cfg["Newtonian_noise"]["time_window"]["tmin"]
    tmax = cfg["Newtonian_noise"]["time_window"]["tmax"]
    
    
    # ---------------------------
    # Plot NN-forces
    # ---------------------------
    
    for mirror in range(mirror_count):
        fig, ax = plt.subplots()
        
        ax.plot(time, forces_bulk[mirror], label="bulk NN", color="tab:blue")
        ax.plot(time, forces_dipole[mirror], label="dipole NN", color="tab:orange")
        ax.plot(time, 12*pi/3 * G*rho*M * loc_disp[mirror], label="bulk prediction", color="tab:green", linestyle="--")
        ax.plot(time, 8*pi/3 * G*rho*M * loc_disp[mirror], label="bulk+cavern prediction", color="tab:red", linestyle="-.")
        ax.plot(time, -4*pi/3 * G*rho*M * loc_disp[mirror], label="cavern prediction", color="tab:purple", linestyle="dotted")
        
        ax.set_xlabel(r"time $t$ [s]")
        ax.set_xlim(tmin, tmax)
        ax.set_ylabel("NN force $F_M$ [N]")
        
        ax.legend(loc="lower center", bbox_to_anchor=(0.45, 0))
        fig.tight_layout()
        fig.savefig(fig_dir / "NN_forces.svg")
    
    endtime = systime.time()
    print("Finished in " + str(np.round((endtime - starttime)/60, 2)) + " min")
    

if __name__ == "__main__":
    main()