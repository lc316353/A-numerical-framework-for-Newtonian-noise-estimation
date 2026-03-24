# -*- coding: utf-8 -*-
"""
05-calc_NN.py

Calculate the Newtonian noise.

Inputs
- config.yaml 
- output_dir/derived/[displacement/curl/div].h5

Outputs
- output_dir/results/[project_name]_[NN_tag]/[NN_data].npy

Date: 11 Mar 2026
Author: Patrick Schillings (patrick.schillings@rwth-aachen.de)
"""

import yaml
from pathlib import Path
import numpy as np
import xarray as xr
import scipy.constants as co
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
    result_dir = out_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    dis_path = derived_dir / "displacement.h5"
    div_path = derived_dir / "div.h5"
    curl_path = derived_dir / "curl.h5"
    
    tag_dir = result_dir / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    
    
    # ---------------------------
    # Load and define parameters
    # ---------------------------
    dataset = cfg["project"]["name"]
    tag=dataset+str(cfg["Newtonian_noise"]["tag"])

    # === Constants ===
    pi = np.pi
    G = co.gravitational_constant
    M = cfg["Newtonian_noise"]["test_mass"]["M"]
    vp = cfg["materials"]["crust"]["vp_m_s"]
    
    if bool(cfg["materials"]["crust"]["rho_from_brocher"]):
        rho = brocher_rho_from_vp(vp)
    else:
        rho = float(cfg["materials"]["crust"]["rho_kg_m3"])
        
    # === Integration domain ===
    dx = cfg["Newtonian_noise"]["integration_domain"]["dx"]
    integration_radius = cfg["Newtonian_noise"]["integration_domain"]["integration_radius"]
    cavern_radius = cfg["Newtonian_noise"]["integration_domain"]["cavern_radius"]
    y_surface = 0
    
    # === Test mass geometry ===
    mirror_positions = np.array(cfg["Newtonian_noise"]["test_mass"]["location"])
    mirror_directions = np.array(cfg["Newtonian_noise"]["test_mass"]["orientation"])
    mirror_count = len(mirror_positions)
    
    if len(mirror_positions[0]) == 2:
        mirror_positions = np.concatenate((mirror_positions.T,np.zeros((1,mirror_count)))).T
    if len(mirror_directions[0]) == 2:
        mirror_directions = np.concatenate((mirror_directions.T,np.zeros((1,mirror_count)))).T
    
    force_const=rho * G * M * dx**3
    
    
    # ---------------------------
    # Load wave field
    # ---------------------------
    
    # === Define coordinates ===
    grid = cfg["decompose"]["grid"]
    xmin, xmax = grid["xmin"], grid["xmax"]
    ymin, ymax = grid["ymin"], grid["ymax"]
    zmin, zmax = -integration_radius, integration_radius
    
    x = np.arange(xmin, xmax + 0.5 * dx, dx)
    y = np.arange(ymin, ymax + 0.5 * dx, dx)
    z = np.arange(zmin, zmax + 0.5 * dx, dx)
    x3d, y3d, z3d = np.meshgrid(x, y, z)
    
    x2d, y2d = np.meshgrid(x, y, indexing="xy")
    coordinates=np.array([x2d.flatten(),y2d.flatten()]).T
        
    # === Load wavefields ===
    wf_ds = xr.load_dataset(dis_path, engine="h5netcdf")
    wf = wf_ds["__xarray_dataarray_variable__"]
    
    div = xr.load_dataset(div_path, engine="h5netcdf")["__xarray_dataarray_variable__"]
    curl = xr.load_dataset(curl_path, engine="h5netcdf")["__xarray_dataarray_variable__"]
    
    x_displacement = wf.isel(c=0).to_numpy()
    y_displacement = wf.isel(c=1).to_numpy()
    div_displacement = div.isel(c=0).to_numpy()
    curl_displacement = curl.isel(c=0).to_numpy()
    
    # ---------------------------
    # Prepare integration domain
    # ---------------------------
    
    # === Time domain ===
    tmax = np.array(wf["t"])[-1]
    tmin = np.array(wf["t"])[0]
    Nt = len(np.array(wf["t"]))
    dt = (tmax - tmin)/(Nt - 1)
    
    tw = cfg["Newtonian_noise"]["time_window"]
    
    from_time_step = int((tw["tmin"] - tmin)/dt)
    to_time_step = int((tw["tmax"] - tmin)/dt)
    
    # === Spatial domain ===
    cavity_kernel = y3d < y_surface
    r3ds = []
    e_rr0 = []
    geo_facts = []
    
    for mirror in range(mirror_count):
        pos = mirror_positions[mirror]
        di = mirror_directions[mirror]
        r3ds.append(np.sqrt((x3d-pos[0])**2 + (y3d-pos[1])**2 + (z3d-pos[2])**2) + 1e-20)
        e_rr0.append(np.array([x3d-pos[0], y3d-pos[1], z3d-pos[2]])/r3ds[mirror])
        cavity_kernel *= r3ds[mirror] > cavern_radius
        geo_facts.append(((x3d-pos[0])*di[0] + (y3d-pos[1])*di[1] + (z3d-pos[2])*di[2])/r3ds[mirror]**3)
    
    for mirror in range(mirror_count):
        geo_facts[mirror] *= cavity_kernel
        geo_facts[mirror] *= r3ds[mirror] < integration_radius
        
        
    # ---------------------------
    # Calculate NN-Forces (t)
    # ---------------------------
    
    # === Force functions ===
    def calc_bulk_force(drho, mirror):
        F = force_const * np.nansum(geo_facts[mirror] * drho)
        return F
    
    def calc_dipole_force(xi_x, xi_y, xi_z, mirror):
        di = np.array(mirror_directions[mirror])
        term1 = di[0]*xi_x + di[1]*xi_y + di[2]*xi_z
        term2 = -3*(xi_x*e_rr0[mirror][0] + xi_y*e_rr0[mirror][1] + xi_z*e_rr0[mirror][2]) * np.einsum("i,ijkl->jkl",di,e_rr0[mirror])
        F = force_const * np.nansum(cavity_kernel/r3ds[mirror]**3 * (term1 + term2))
        return F
    
    # === Result container ===
    forces_bulk = np.zeros((mirror_count, Nt))
    forces_dipole = np.zeros((mirror_count, Nt))
    
    loc_x_disp = np.zeros((mirror_count, Nt))
    loc_y_disp = np.zeros((mirror_count, Nt))
    loc_div = np.zeros((mirror_count, Nt))
    loc_curl = np.zeros((mirror_count, Nt))
    
    for t in range(from_time_step,to_time_step):
        
        ordered_x_displacement = x_displacement[t].reshape(len(x), len(y))
        ordered_y_displacement = y_displacement[t].reshape(len(x), len(y))
        ordered_div_displacement = div_displacement[t].reshape(len(x), len(y))
        ordered_curl_displacement = curl_displacement[t].reshape(len(x), len(y))
        
        # === Calculate forces ===
        for mirror in range(mirror_count):
            forces_bulk[mirror][t] = calc_bulk_force(-np.moveaxis(np.stack([ordered_div_displacement]*len(z)),0,-1), mirror)
            forces_dipole[mirror][t] = calc_dipole_force(np.moveaxis(np.stack([ordered_x_displacement]*len(z)),0,-1), np.moveaxis(np.stack([ordered_y_displacement]*len(z)),0,-1), 0, mirror)
            loc_x_disp[mirror][t] = ordered_x_displacement[int((mirror_positions[mirror][0]-xmin)//dx), int((mirror_positions[mirror][1]-ymin)//dx)]
            loc_y_disp[mirror][t] = ordered_y_displacement[int((mirror_positions[mirror][0]-xmin)//dx), int((mirror_positions[mirror][1]-ymin)//dx)]
            loc_div[mirror][t] = ordered_div_displacement[int((mirror_positions[mirror][0]-xmin)//dx), int((mirror_positions[mirror][1]-ymin)//dx)]
            loc_curl[mirror][t] = ordered_curl_displacement[int((mirror_positions[mirror][0]-xmin)//dx), int((mirror_positions[mirror][1]-ymin)//dx)]

        # === Advancement log ===
        if t%10 == 0:
            print(t*dt + tmin,"/",tmax)
    
    
    # ---------------------------
    # Save results
    # ---------------------------
    
    np.save(tag_dir/"bulk_forces", forces_bulk)
    np.save(tag_dir/"dipole_forces", forces_dipole)
    np.save(tag_dir/"x_displacements", loc_x_disp)
    np.save(tag_dir/"y_displacements", loc_y_disp)
    np.save(tag_dir/"divergences", loc_div)
    np.save(tag_dir/"curls", loc_curl)
    
    cfg_clone_path = tag_dir / "config.yaml"
    with cfg_clone_path.open("w") as f:
        yaml.safe_dump(cfg, f, indent=4, sort_keys=False)
    
    endtime = systime.time()
    print(f"Wrote forces and local displacements: {tag_dir}")
    print("Finished in " + str(np.round((endtime - starttime)/60, 2)) + " min")

if __name__ == "__main__":
    main()