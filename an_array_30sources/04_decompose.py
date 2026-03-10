#!/usr/bin/env python3
"""
04_decompose.py

Extract displacement and gradient-of-displacement from Salvus volume output,
sample on a user-defined regular 2-D grid, window in time, and save:
  - displacement
  - divergence (∂ux/∂x + ∂uy/∂y)
  - scalar curl (∂uy/∂x − ∂ux/∂y)

Inputs
- config.yaml (must be in the same directory as this script)
- output/sim/volume_output.h5 (simulation.volume_output.filename)

Outputs
- defined by decompose.outputs.* in config.yaml

Notes
-----
2-D assumption and gradient component order:
  c=0: ∂ux/∂x
  c=1: ∂ux/∂y
  c=2: ∂uy/∂x
  c=3: ∂uy/∂y

Date: 10 Mar 2026
Author: Shi Yao (yaoshi229@gmail.com)
"""

from pathlib import Path

import numpy as np
import yaml
from salvus.toolbox.helpers.wavefield_output import WavefieldOutput, wavefield_output_to_xarray


def main():
    # ---------------------------
    # Load config.yaml (local)
    # ---------------------------
    cfg_path = Path(__file__).with_name("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find {cfg_path}. Put config.yaml in the same directory as this script.")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["paths"]["output_dir"])
    sim_dir = out_dir / "sim"
    derived_dir = out_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Input wavefield file
    # ---------------------------
    wavefield_path = sim_dir / cfg["simulation"]["volume_output"]["filename"]
    if not wavefield_path.exists():
        raise FileNotFoundError(f"Missing wavefield file: {wavefield_path}. Run 03_run_simulation.py first.")

    # ---------------------------
    # Decompose parameters (config.yaml)
    # ---------------------------
    dec = cfg["decompose"]

    tw = dec["time_window"]
    tmin = float(tw["tmin_s"])
    tmax = float(tw["tmax_s"])

    grid = dec["grid"]
    xmin = float(grid["xmin"])
    xmax = float(grid["xmax"])
    ymin = float(grid["ymin"])
    ymax = float(grid["ymax"])
    interval = float(grid["interval"])

    outs = dec["outputs"]
    out_disp = Path(outs["displacement"])
    out_div = Path(outs["div"])
    out_curl = Path(outs["curl"])

    out_disp.parent.mkdir(parents=True, exist_ok=True)
    out_div.parent.mkdir(parents=True, exist_ok=True)
    out_curl.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Build query points (regular grid)
    # ---------------------------
    x = np.arange(xmin, xmax + 0.5 * interval, interval)
    y = np.arange(ymin, ymax + 0.5 * interval, interval)
    X, Y = np.meshgrid(x, y, indexing="xy")
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # ---------------------------
    # Displacement (time-windowed)
    # ---------------------------
    displacement = WavefieldOutput.from_file(str(wavefield_path), field="displacement", output_type="volume")
    u = wavefield_output_to_xarray(displacement, points=points).sel(t=slice(tmin, tmax))
    u.to_netcdf(out_disp, engine="h5netcdf")

    # ---------------------------
    # Gradient -> div + curl (time-windowed)
    # ---------------------------
    grad = WavefieldOutput.from_file(str(wavefield_path), field="gradient-of-displacement", output_type="volume")
    g = wavefield_output_to_xarray(grad, points=points).sel(t=slice(tmin, tmax))

    if "c" not in g.dims:
        raise ValueError("Expected gradient field to have a component dimension named 'c'.")
    if g.sizes["c"] < 4:
        raise ValueError(f"Expected at least 4 gradient components for 2-D, got {g.sizes['c']}.")

    div_u = (g.sel(c=0) + g.sel(c=3)).expand_dims(dim={"c": [0]})
    curl_u = (g.sel(c=2) - g.sel(c=1)).expand_dims(dim={"c": [0]})

    div_u.to_netcdf(out_div, engine="h5netcdf")
    curl_u.to_netcdf(out_curl, engine="h5netcdf")

    print(f"Input wavefield:     {wavefield_path}")
    print(f"Time window:         [{tmin:.2f}, {tmax:.2f}] s")
    print(f"Grid:                x[{xmin:.1f},{xmax:.1f}], y[{ymin:.1f},{ymax:.1f}], dx=dy={interval:.1f}")
    print(f"Wrote displacement:  {out_disp}")
    print(f"Wrote div(u):        {out_div}")
    print(f"Wrote curl(u):       {out_curl}")


if __name__ == "__main__":
    main()
