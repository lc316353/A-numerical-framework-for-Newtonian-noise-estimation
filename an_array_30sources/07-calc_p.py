# -*- coding: utf-8 -*-
"""
07-calc_p.py

Calculates the P-wave fraction p.

Inputs
- config.yaml 
- output_dir/derived/[displacement/curl/div].h5

Outputs
- fig_dir/NN_results[project_name]_[NN_tag]/[p_plots].svg
- output_dir/results/[project_name]_[NN_tag]/[p_data].txt

Date: 11 Mar 2026
Author: Patrick Schillings (patrick.schillings@rwth-aachen.de)
"""

import yaml
from pathlib import Path
import xarray as xr
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy.constants as co
import time as systime
import scipy.signal as sig

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
    div_path = derived_dir / "div.h5"
    curl_path = derived_dir / "curl.h5"
    
    result_dir = out_dir / "results"
    tag_dir = result_dir / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    
    fig_dir = Path(cfg["paths"]["fig_dir"]) / "NN_results" / tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    
    # ---------------------------
    # Load and define parameters
    # ---------------------------

    # === Constants ===
    pi = np.pi
    c_p = cfg["materials"]["crust"]["vp_m_s"]
    c_s = c_p/cfg["materials"]["crust"]["vp_vs_ratio"]
    
    # === Grid ===
    cfgp = cfg["Newtonian_noise"]["p_fraction"]
    fromx, tox = cfgp["x_bounds"]
    fromy, toy = cfgp["y_bounds"]
    points = cfgp["points"]
    
    # ---------------------------
    # Load wave field
    # ---------------------------
    
    # === Define coordinates ===
    grid = cfg["decompose"]["grid"]
    xmin, xmax = grid["xmin"], grid["xmax"]
    ymin, ymax = grid["ymin"], grid["ymax"]
    
    dx = cfg["Newtonian_noise"]["integration_domain"]["dx"]
    x = np.arange(xmin, xmax + 0.5 * dx, dx)
    y = np.arange(ymin, ymax + 0.5 * dx, dx)
    
    x2d, y2d = np.meshgrid(x, y, indexing="xy")
    coordinates=np.array([x2d.flatten(),y2d.flatten()]).T
    
    x = coordinates[:, 0]
    y = coordinates[:, 1]
        
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
    # Prepare time domain
    # ---------------------------
    
    # === Time ===
    tmax = np.array(wf["t"])[-1]
    tmin = np.array(wf["t"])[0]
    Nt = len(np.array(wf["t"]))
    dt = (tmax - tmin)/(Nt - 1)
    
    # === Frequencies ===
    nperseg = (Nt - 1) // 7.5
    
    fmin, fmax = cfg["sources"]["stf"]["fmin_hz"], cfg["sources"]["stf"]["f_taper_start_hz"]
    
    freq_x = sig.welch(curl_displacement[:,0], 1/dt, nperseg=nperseg)[0]
    
    fmin_ind, fmax_ind = np.argmax(freq_x>fmin), np.argmax(freq_x>fmax)
    freq_x = freq_x[fmin_ind:fmax_ind]
    
    
    # ---------------------------
    # Find suitable mesh points
    # ---------------------------
    
    # === Prepare grid ===
    xinter = np.arange(fromx, tox+1, (tox - fromx) // (points-1))
    yinter = np.arange(fromy, toy+1, (toy - fromy) // (points-1))
    xin, yin = np.meshgrid(xinter, yinter)
    
    mirror_positions = np.array([xin.flatten(), yin.flatten(), np.zeros(len(xinter) * len(yinter))]).T
    mirror_positions = np.array(mirror_positions)
    mirror_count = len(mirror_positions)
    
    # === Prepare containers ===
    loc_x_disp = np.zeros((mirror_count, Nt))
    loc_y_disp = np.zeros((mirror_count, Nt))
    loc_div = np.zeros((mirror_count, Nt))
    loc_curl = np.zeros((mirror_count, Nt))
    loc_p_dc = np.zeros((mirror_count, len(freq_x)))
    p_mean = np.zeros((mirror_count))
    loc_test_frac = np.zeros((mirror_count, len(freq_x)))
    
    # === Match grid to actual mesh points ===
    accuracy = dx/2
    
    for mirror in range(mirror_count):
        xlower = mirror_positions[mirror][0] - accuracy
        xupper = mirror_positions[mirror][0] + accuracy
        ylower = mirror_positions[mirror][1] - accuracy
        yupper = mirror_positions[mirror][1] + accuracy
        
        coordinate_index = np.argmax((x>xlower) * (x<xupper) * (y>ylower) * (y<yupper))
        
        if coordinate_index == 0:
            print("WARNING: Accuracy might be too low, index returned 0 for mirror " + str(mirror))
        
        
        # ---------------------------
        # Calculate P-fraction
        # ---------------------------
        
        # === Fetch local displacements ===
        loc_x_disp[mirror] = x_displacement[:, coordinate_index]
        loc_y_disp[mirror] = y_displacement[:, coordinate_index]
        loc_div[mirror] = div_displacement[:, coordinate_index]
        loc_curl[mirror] = curl_displacement[:, coordinate_index]
        
        # === PSDs ===
        loc_x_disp_PSD = sig.welch(x_displacement[:,coordinate_index], 1/dt, nperseg=nperseg)[1][fmin_ind:fmax_ind]
        loc_y_disp_PSD = sig.welch(y_displacement[:,coordinate_index], 1/dt, nperseg=nperseg)[1][fmin_ind:fmax_ind]
        loc_div_PSD = sig.welch(div_displacement[:, coordinate_index], 1/dt, nperseg=nperseg)[1][fmin_ind:fmax_ind]
        loc_curl_PSD = sig.welch(curl_displacement[:, coordinate_index], 1/dt, nperseg=nperseg)[1][fmin_ind:fmax_ind]
        
        # === P-wave fraction ===
        loc_p_dc[mirror] = loc_div_PSD / ((c_s/c_p)**2 * loc_curl_PSD + loc_div_PSD)
        p_mean[mirror] = np.mean(loc_p_dc[mirror])
        
        # === Test total displacement equality ===
        loc_test_frac[mirror] = (loc_x_disp_PSD + loc_y_disp_PSD)/((2*pi*freq_x/c_p)**(-2) * loc_div_PSD + (2*pi*freq_x/c_s)**(-2) * loc_curl_PSD)
        
        
    # ---------------------------
    # Plotting
    # ---------------------------
    
    fig,ax= plt.subplots()
    color = 1 - ((mirror_positions - np.min(mirror_positions, axis=0)) / np.concatenate([2 * (np.max(mirror_positions, axis=0) - np.min(mirror_positions, axis=0))[:2], [1]]) + np.array([0.25, 0.25, 0]))
    for mirror in range(mirror_count):
        ax.plot(freq_x, loc_p_dc[mirror], color=color[mirror], alpha=0.075)
        ax.axhline(p_mean[mirror], color=color[mirror], linestyle="dashed", alpha=0.125)
    
    ax.plot([], [], "k-", label=r"$p$ at 100 different positions", alpha=0.075)
    ax.plot([], [], "k--", label=r"frequency-averages at these positions", alpha=0.125)
    ax.axhline(np.median(p_mean), linestyle="--", color="k", label="median of frequency averages")
    ax.axhline(np.quantile(p_mean, 0.16), linestyle=":", color="k", label="16% and 84% quantiles")
    ax.axhline(np.quantile(p_mean, 0.84), linestyle=":", color="k")
    
    ax.set_xlabel(r"frequency $f$ [Hz]")
    ax.set_ylabel(r"P-wave fraction $p$")
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 0.5)
    
    ax.legend()
    fig.savefig(fig_dir / ("p_" + str(int((tox+fromx)/2)) + "_" + str(int((toy+fromy)/2)) + ".svg"))
            
    # === Prints ===
    print("Consistency test:\nmean((<xi_x^2>+<xi_y^2>)/(k_P^{-2}<div(xi)^2>+k_S<curl(xi)^2>) = " + str(np.mean(loc_test_frac)) + " +- " + str(np.std(loc_test_frac) / np.sqrt(len(loc_test_frac))))
    print("\ntotal median p: " + str(np.median(p_mean)) + " +- " + str(np.quantile(p_mean,0.84)-np.quantile(p_mean, 0.16))+"\n")

    
    endtime = systime.time()
    print("Finished in " + str(np.round((endtime - starttime)/60, 2)) + " min")

if __name__ == "__main__":
    main()
