#!/usr/bin/env python3
"""
06_decompose.py

Compute displacement, divergence, and curl from Salvus volume wavefield outputs.

Inputs
- config.yaml (must be in the same directory as this script)
- simulation.volume_output.filename

Outputs (under paths.output_dir/derived/)
- displacement.nc  : displacement (xarray -> NetCDF/HDF5)
- div.nc           : divergence of displacement
- curl.nc          : 2D scalar curl (out-of-plane component)

Date: 10 Mar 2026
Author: Shi Yao (yaoshi229@gmail.com)
"""

from pathlib import Path

import h5py
import numpy as np
import yaml
from salvus.toolbox.helpers.wavefield_output import WavefieldOutput, wavefield_output_to_xarray


def main():
    # ---------------------------
    # Load config.yaml (local)
    # ---------------------------
    cfg_path = Path(__file__).with_name("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Cannot find {cfg_path}. Put config.yaml in the same directory as this script."
        )
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["paths"]["output_dir"])
    sim_dir = out_dir / "sim"
    derived_dir = out_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Input wavefield file (from existing config keys)
    # ---------------------------
    wavefield_path = sim_dir / cfg["simulation"]["volume_output"]["filename"]
    if not wavefield_path.exists():
        raise FileNotFoundError(
            f"Missing wavefield file: {wavefield_path}. Run 03_run_simulation.py first."
        )

    # Coordinate dataset name
    coord_dataset = cfg.get("decompose", {}).get("coordinate_dataset", "coordinates_ELASTIC")

    # ---------------------------
    # Output files (keep simple; no extra config keys required)
    # ---------------------------
    out_disp = derived_dir / "displacement.nc"
    out_div = derived_dir / "div.nc"
    out_curl = derived_dir / "curl.nc"

    # ---------------------------
    # Build point list from structured coordinates
    # ---------------------------
    with h5py.File(wavefield_path, "r") as f:
        coordinates = f[coord_dataset][:]

    n1, n2, _ = coordinates.shape
    x = coordinates[:, :, 0].reshape(n1 * n2)
    y = coordinates[:, :, 1].reshape(n1 * n2)
    points = np.stack([x, y], axis=-1)

    # ---------------------------
    # Displacement
    # ---------------------------
    displacement = WavefieldOutput.from_file(str(wavefield_path), field="displacement", output_type="volume")
    displacement_xr = wavefield_output_to_xarray(displacement, points=points)
    displacement_xr.to_netcdf(out_disp, engine="h5netcdf")

    # ---------------------------
    # Gradient of displacement -> div + curl
    # ---------------------------
    grad = WavefieldOutput.from_file(
        str(wavefield_path), field="gradient-of-displacement", output_type="volume"
    )
    grad_xr = wavefield_output_to_xarray(grad, points=points)

    # Keep your original indexing convention:
    # div(u) = du_x/dx + du_y/dy -> c=0 and c=3
    # curl_z = du_y/dx - du_x/dy -> c=2 - c=1
    div_u = (grad_xr.sel(c=0) + grad_xr.sel(c=3)).expand_dims(dim={"c": [0]})
    curl_u = (grad_xr.sel(c=2) - grad_xr.sel(c=1)).expand_dims(dim={"c": [0]})

    div_u.to_netcdf(out_div, engine="h5netcdf")
    curl_u.to_netcdf(out_curl, engine="h5netcdf")

    print(f"Input wavefield:     {wavefield_path}")
    print(f"Wrote displacement:  {out_disp}")
    print(f"Wrote div(u):        {out_div}")
    print(f"Wrote curl(u):       {out_curl}")


if __name__ == "__main__":
    main()
    