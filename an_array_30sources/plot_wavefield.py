#!/usr/bin/env python3
"""
plot_wavefield.py

Inputs
- config.yaml
- decompose.outputs.displacement
- decompose.outputs.div
- decompose.outputs.curl
- decompose.time_window (tmin_s, tmax_s)
- decompose.grid (xmin/xmax/ymin/ymax/interval)

Outputs
- figures/wavefields_window_grid/*.png

Date: 10 Mar 2026
Author: Shi Yao (yaoshi229@gmail.com)
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import yaml


def load_first_dataarray(ds_path: str) -> xr.DataArray:
    ds = xr.load_dataset(ds_path, engine="h5netcdf")
    varname = list(ds.data_vars)[0]
    return ds[varname]

def maybe_time_slice(da: xr.DataArray, tmin, tmax) -> xr.DataArray:
    if "t" not in da.coords:
        raise ValueError(f"'t' coordinate not found in {da.name or 'DataArray'}")
    if tmin is None and tmax is None:
        return da
    if tmin is None:
        return da.sel(t=slice(None, tmax))
    if tmax is None:
        return da.sel(t=slice(tmin, None))
    return da.sel(t=slice(tmin, tmax))


def main():
    # ------------------ Load config ------------------
    cfg_path = Path(__file__).with_name("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find {cfg_path}. Put config.yaml in the same directory as this script.")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    # ------------------ Paths from config ------------------
    dec = cfg["decompose"]
    outs = dec["outputs"]

    dis_path = Path(outs["displacement"])
    div_path = Path(outs["div"])
    curl_path = Path(outs["curl"])

    for p in [dis_path, div_path, curl_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}. Run 04_decompose.py first.")

    # ------------------ Grid from config ------------------
    grid = dec["grid"]
    xmin, xmax = float(grid["xmin"]), float(grid["xmax"])
    ymin, ymax = float(grid["ymin"]), float(grid["ymax"])
    interval = float(grid["interval"])

    x1d = np.arange(xmin, xmax + 0.5 * interval, interval)
    y1d = np.arange(ymin, ymax + 0.5 * interval, interval)
    X, Y = np.meshgrid(x1d, y1d, indexing="xy")
    x = X.ravel()
    y = Y.ravel()

    # ------------------ Time window from config ------------------
    tw = dec["time_window"]
    tmin = float(tw["tmin_s"])
    tmax = float(tw["tmax_s"])

    # ------------------ Plot settings (keep local) ------------------
    fig_dir = Path("./figures/wavefields")
    fig_dir.mkdir(parents=True, exist_ok=True)

    frame_stride = 1
    save_every = 100
    use_fixed_extent = True

    vmin_disp, vmax_disp = -1e-17, 1e-17
    vmin_derive, vmax_derive = -1e-19, 1e-19

    # ------------------ Load fields ------------------
    wf = load_first_dataarray(str(dis_path))
    div = load_first_dataarray(str(div_path))
    curl = load_first_dataarray(str(curl_path))

    # Enforce time window from config (no "maybe")
    wf   = maybe_time_slice(wf,   tmin, tmax)
    div  = maybe_time_slice(div,  tmin, tmax)
    curl = maybe_time_slice(curl, tmin, tmax)

    t_vals = wf.coords["t"].values[::frame_stride]
    # print(len(t_vals))

    # ------------------ Plot loop ------------------
    for i_t, tval in enumerate(t_vals):
        if i_t % save_every != 0:
            continue

        t_idx = int(np.searchsorted(wf.coords["t"].values, tval))

        z_ux = wf.isel(t=t_idx, c=0).values
        z_uy = wf.isel(t=t_idx, c=1).values
        z_div = div.isel(t=t_idx, c=0).values
        z_curl = curl.isel(t=t_idx, c=0).values

        fig, axs = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

        im0 = axs[0, 0].scatter(x, y, c=z_ux, cmap="seismic", s=5, vmin=vmin_disp, vmax=vmax_disp)
        axs[0, 0].set_title("Wavefield [ux]")
        axs[0, 0].set_aspect("equal")

        im1 = axs[0, 1].scatter(x, y, c=z_uy, cmap="seismic", s=5, vmin=vmin_disp, vmax=vmax_disp)
        axs[0, 1].set_title("Wavefield [uy]")
        axs[0, 1].set_aspect("equal")

        im2 = axs[1, 0].scatter(x, y, c=z_div, cmap="seismic", s=5, vmin=vmin_derive, vmax=vmax_derive)
        axs[1, 0].set_title("P wave [div(u)]")
        axs[1, 0].set_aspect("equal")

        im3 = axs[1, 1].scatter(x, y, c=z_curl, cmap="seismic", s=5, vmin=vmin_derive, vmax=vmax_derive)
        axs[1, 1].set_title("S wave [curl(u)]")
        axs[1, 1].set_aspect("equal")

        if use_fixed_extent:
            for ax in axs.ravel():
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])

        for ax in axs.ravel():
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        fig.colorbar(im1, ax=axs[0, 1], shrink=0.8)
        fig.colorbar(im3, ax=axs[1, 1], shrink=0.8)

        fig.suptitle(f"Time: {float(tval):.3f} s", fontsize=16)

        out_name = fig_dir / f"{i_t:05d}.png"
        fig.savefig(out_name, dpi=150)
        plt.close(fig)

    print(f"Finished saving wavefield plots for t in [{tmin}, {tmax}] s.")


if __name__ == "__main__":
    main()

