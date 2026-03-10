#!/usr/bin/env python3
"""
03_run_simulation.py

Project 2: Run a 2D Salvus waveform simulation with multiple random sources.

Inputs
- config.yaml
- Mesh: mesh.mesh_file
- Observations: paths.obs_json 
- STFs: sources.stf.output_h5 (HDF5 with stf_1, stf_2, ...)

Outputs (under paths.output_dir/sim/):
- receivers.h5
- volume_output.h5

Date: 10 Mar 2026
Author: Shi Yao (yaoshi229@gmail.com)
"""

from pathlib import Path
import json

import h5py
import yaml
from salvus.mesh import UnstructuredMesh
import salvus.namespace as sn


def _load_stfs(h5_path: Path):
    """Return a list of STF arrays shaped (N, 1), ordered by key name."""
    with h5py.File(h5_path, "r") as h5file:
        keys = sorted(h5file.keys())
        stfs = [h5file[k][:][:, None] for k in keys]  # (N,) -> (N, 1)
    return stfs, keys


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
    # Paths
    # ---------------------------
    out_dir = Path(cfg["paths"]["output_dir"])
    fig_dir = Path(cfg["paths"]["fig_dir"])
    sim_dir = out_dir / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(cfg["mesh"]["mesh_file"])
    obs_path = Path(cfg["paths"]["obs_json"])
    stf_h5 = Path(cfg["sources"]["stf"]["output_h5"])

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}. Run 02_build_mesh.py first.")
    if not obs_path.exists():
        raise FileNotFoundError(f"Obs not found: {obs_path}. Run 01_make_obs.py first.")
    if not stf_h5.exists():
        raise FileNotFoundError(f"STF file not found: {stf_h5}. Run 00_gen_stfs_taper.py first.")

    # ---------------------------
    # Load mesh + obs + stfs
    # ---------------------------
    mesh = UnstructuredMesh.from_h5(str(mesh_path))

    with obs_path.open("r") as f:
        obs_dict = json.load(f)

    src_keys = list(obs_dict["sources"].keys())
    stf_list, stf_keys = _load_stfs(stf_h5)

    ns_cfg = int(cfg["sources"]["nsources"])
    if len(src_keys) != ns_cfg:
        raise ValueError(f"Mismatch: sources.nsources={ns_cfg}, but obs.json has {len(src_keys)} sources.")
    if len(stf_list) != ns_cfg:
        raise ValueError(f"Mismatch: sources.nsources={ns_cfg}, but stf.h5 has {len(stf_list)} datasets.")

    # ---------------------------
    # Build sources: locations + custom STF
    # ---------------------------
    stf_cfg = cfg["sources"]["stf"]
    fs_hz = float(stf_cfg["fs_hz"])
    start_time_shift = 1.0  # keep same as your script

    src_list = []
    for i, src_key in enumerate(src_keys):
        sx = float(obs_dict["sources"][src_key]["x"])
        sy = float(obs_dict["sources"][src_key]["y"])
        stf_arr = stf_list[i]

        source_time_function = sn.simple_config.stf.Custom.from_array(
            array=stf_arr,
            sampling_rate_in_hertz=fs_hz,
            start_time_in_seconds=start_time_shift,
            dataset_name="/stf",
        )

        src = sn.simple_config.source.cartesian.SideSetVectorPoint2D(
            point=(sx, sy),
            direction=(0, 1),
            side_set_name="y1",
            fx=0.0,
            fy=1.0,
            offset=0.0,
            source_time_function=source_time_function,
        )
        src_list.append(src)

    # ---------------------------
    # Receivers
    # ---------------------------
    rec_list = []
    for sta, sta_dict in obs_dict["receivers"].items():
        x = float(sta_dict["x"])
        y = float(sta_dict["y"])
        net = sta.split(".")[0]
        sta_code = sta.split(".")[1]

        if abs(y) < 1e-12:
            rec = sn.simple_config.receiver.cartesian.SideSetPoint2D(
                point=(x, y),
                direction=(0, 1),
                side_set_name="y1",
                fields=["velocity", "displacement"],
                network_code=net,
                station_code=sta_code,
            )
        else:
            rec = sn.simple_config.receiver.cartesian.Point2D(
                x=x,
                y=y,
                fields=["velocity", "displacement"],
                network_code=net,
                station_code=sta_code,
            )
        rec_list.append(rec)

    # ---------------------------
    # Absorbing boundary (from config)
    # ---------------------------
    abc = cfg["absorbing_boundary"]
    absorbing = sn.simple_config.boundary.Absorbing(
        width_in_meters=float(abc["width_m"]),
        side_sets=list(abc["side_sets"]),
        taper_amplitude=float(abc["taper_amplitude"]),
    )

    # ---------------------------
    # Simulation time + outputs (mostly from config)
    # ---------------------------
    tcfg = cfg["simulation"]["time"]
    time_step = float(tcfg["dt"])
    end_time = float(tcfg.get("t1", 31.0))
    start_time = float(tcfg.get("t0", 0.0))

    vol_cfg = cfg["simulation"]["volume_output"]
    sampling_interval = int(vol_cfg["sampling_interval_steps"])

    w = sn.simple_config.simulation.Waveform(
        mesh=mesh,
        sources=src_list,
        receivers=rec_list,
        start_time_in_seconds=start_time,
        end_time_in_seconds=end_time,
    )
    w.physics.wave_equation.time_step_in_seconds = time_step
    w.physics.wave_equation.boundaries = [absorbing]

    w.output.volume_data.format = str(vol_cfg["format"])
    w.output.volume_data.fields = list(vol_cfg["fields"])
    w.output.volume_data.filename = str(vol_cfg["filename"])
    w.output.volume_data.sampling_interval_in_time_steps = sampling_interval

    sim_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Run
    # ---------------------------
    simcfg = cfg["simulation"]
    sn.api.run(
        input_file=w,
        site_name=str(simcfg["site_name"]),
        output_folder=str(sim_dir),
        overwrite=True,
        delete_remote_files=True,
        wall_time_in_seconds=int(simcfg["wall_time_s"]),
        ranks=int(simcfg["ranks"]),
        get_all=True,
    )

    print(f"Mesh:      {mesh_path}")
    print(f"Obs:       {obs_path}")
    print(f"STF:       {stf_h5} ({len(stf_list)} datasets)")
    print(f"Output folder: {sim_dir}")

if __name__ == "__main__":
    main()
