#!/usr/bin/env python3
"""
plot_wavefield.py

Plot the displacement wavefield, divergence and curl.

Inputs
- output/sim/volume_output.h5
- output/derived/displacement.nc
- output/derived/div.nc
- output/derived/curl.nc

Outputs
- figures/wavefields/*.png

Date: 10 Mar 2026
Author: Shi Yao (yaoshi229@gmail.com)
"""

from pathlib import Path

import h5py
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ------------------------- params -------------------------
xmin, xmax = 0, 20000
ymin, ymax = -8000, 0

# Salvus time step and volume output sampling interval used during simulation
time_step = 1e-3
sampling_interval_in_time_steps = 10

# Project paths (fixed)
out_dir = Path("./output")
wavefield_path = out_dir / "sim" / "volume_output.h5"
derived_dir = out_dir / "derived"
dis_path = derived_dir / "displacement.nc"
div_path = derived_dir / "div.nc"
curl_path = derived_dir / "curl.nc"

# Figure output
fig_dir = Path("./figures/wavefields")
fig_dir.mkdir(parents=True, exist_ok=True)

# Plot control
plot_every_n_steps = 100

# Color scales
vmin_disp, vmax_disp = -1e-6, 1e-6
vmin_derive, vmax_derive = -1e-8, 1e-8
# ----------------------------------------------------------


def load_dataarray_from_nc(path: Path) -> xr.DataArray:
    """Load the single DataArray stored in a NetCDF/HDF5 file written by xarray."""
    ds = xr.load_dataset(path, engine="h5netcdf")
    return ds["__xarray_dataarray_variable__"]


def main():
    # --- sanity checks ---
    for p in [wavefield_path, dis_path, div_path, curl_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    # --- load point coordinates from Salvus volume output ---
    with h5py.File(wavefield_path, "r") as f:
        coordinates = f["coordinates_ELASTIC"][:]

    n1, n2, _ = coordinates.shape
    x = coordinates[:, :, 0].reshape(n1 * n2)
    y = coordinates[:, :, 1].reshape(n1 * n2)

    # --- load wavefields from decompose outputs ---
    wf = load_dataarray_from_nc(dis_path)    # displacement: t, point, c(=2)
    div = load_dataarray_from_nc(div_path)   # div(u):       t, point, c(=1)
    curl = load_dataarray_from_nc(curl_path) # curl(u):      t, point, c(=1)

    nt = wf.sizes["t"]

    for t in range(0, nt, plot_every_n_steps):
        print(f"Plotting time step {t}/{nt - 1}")

        fig, axs = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)

        # ux
        z0 = wf.isel(t=t, c=0).values
        im0 = axs[0, 0].scatter(x, y, c=z0, cmap="seismic", s=5, vmin=vmin_disp, vmax=vmax_disp)
        axs[0, 0].set_title("Wavefield [ux]")

        # uy
        z1 = wf.isel(t=t, c=1).values
        im1 = axs[0, 1].scatter(x, y, c=z1, cmap="seismic", s=5, vmin=vmin_disp, vmax=vmax_disp)
        axs[0, 1].set_title("Wavefield [uy]")

        # div(u)
        z2 = div.isel(t=t, c=0).values
        im2 = axs[1, 0].scatter(x, y, c=z2, cmap="seismic", s=5, vmin=vmin_derive, vmax=vmax_derive)
        axs[1, 0].set_title("P wave proxy [div(u)]")

        # curl(u)
        z3 = curl.isel(t=t, c=0).values
        im3 = axs[1, 1].scatter(x, y, c=z3, cmap="seismic", s=5, vmin=vmin_derive, vmax=vmax_derive)
        axs[1, 1].set_title("S wave proxy [curl(u)]")

        # axes formatting
        for ax in axs.ravel():
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

        # minimal colorbars (same style as your original)
        fig.colorbar(im1, ax=axs[0, 1], shrink=0.8)
        fig.colorbar(im3, ax=axs[1, 1], shrink=0.8)

        time_s = t * time_step * sampling_interval_in_time_steps
        fig.suptitle(f"Time: {time_s:.3f} s", fontsize=16)

        out_png = fig_dir / f"{t:06d}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    print("Finished saving wavefield plots.")


if __name__ == "__main__":
    main()
