#!/usr/bin/env python3
"""
03_run_simulation.py

Run a 2D Salvus waveform simulation.

Inputs
- Mesh: mesh.mesh_file (from config.yaml)
- Observations: paths.obs_json (source/receiver geometry)
- Source: a single Ricker source (simulation.source)

Outputs (under paths.output_dir/sim/)
- receivers.h5: receiver time series
- volume_output.h5: volume wavefield output

Date: 10 Mar 2026
Author: Shi Yao (yaoshi229@gmail.com)
"""

from pathlib import Path
import json

import yaml
from salvus.mesh import UnstructuredMesh
import salvus.namespace as sn


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
    # Resolve paths
    # ---------------------------
    out_dir = Path(cfg["paths"]["output_dir"])
    fig_dir = Path(cfg["paths"]["fig_dir"])
    obs_path = Path(cfg["paths"]["obs_json"])
    mesh_path = Path(cfg["mesh"]["mesh_file"])

    (out_dir / "sim").mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}. Run 02_build_mesh.py first.")
    if not obs_path.exists():
        raise FileNotFoundError(f"Obs not found: {obs_path}. Run 01_make_obs.py first.")

    # Output folder for this run 
    data_out_dir = out_dir / "sim"
    data_out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Load mesh + obs
    # ---------------------------
    mesh = UnstructuredMesh.from_h5(str(mesh_path))
    with obs_path.open("r") as f:
        obs_dict = json.load(f)

    # ---------------------------
    # Source (single)
    # ---------------------------
    src_key = list(obs_dict["sources"].keys())[0]
    src_x = float(obs_dict["sources"][src_key]["x"])
    src_y = float(obs_dict["sources"][src_key]["y"])

    scfg = cfg["simulation"]["source"]

    source_time_function = sn.simple_config.stf.Ricker(
        center_frequency=float(scfg["center_frequency_hz"]),
        time_shift_in_seconds=float(scfg["time_shift_s"]),
    )

    # Save STF figure
    stf_fig = source_time_function.plot(show=False)
    stf_png = fig_dir / "stf_ricker.png"
    stf_fig.savefig(stf_png, dpi=300, bbox_inches="tight")

    # Setting for source
    src = sn.simple_config.source.cartesian.SideSetVectorPoint2D(
        point=(src_x, src_y),
        direction=tuple(scfg["direction"]),
        side_set_name=str(scfg["side_set_name"]),
        fx=float(scfg["fx"]),
        fy=float(scfg["fy"]),
        offset=0.0,
        source_time_function=source_time_function,
    )

    # ---------------------------
    # Receivers
    # ---------------------------
    rec_list = []
    for sta, sta_dict in obs_dict["receivers"].items():
        x = float(sta_dict["x"])
        y = float(sta_dict["y"])
        net = sta.split(".")[0]
        sta_code = sta.split(".")[1]

        if sta_dict['y'] == 0:
            rec = sn.simple_config.receiver.cartesian.SideSetPoint2D(
                point=(x, y),
                direction=(0, 1),
                side_set_name=str(cfg["simulation"]["source"]["side_set_name"]),
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
    # Absorbing boundary
    # ---------------------------
    abc = cfg["absorbing_boundary"]
    absorbing = sn.simple_config.boundary.Absorbing(
        width_in_meters=float(abc["width_m"]),
        side_sets=list(abc["side_sets"]),
        taper_amplitude=float(abc["taper_amplitude"]),
    )

    # ---------------------------
    # Simulation time + outputs
    # ---------------------------
    tcfg = cfg["simulation"]["time"]
    w = sn.simple_config.simulation.Waveform(
        mesh=mesh,
        sources=src,
        receivers=rec_list,
        start_time_in_seconds=float(tcfg["t0"]),
        end_time_in_seconds=float(tcfg["t1"]),
    )
    w.physics.wave_equation.time_step_in_seconds = float(tcfg["dt"])
    w.physics.wave_equation.boundaries = [absorbing]

    vol = cfg["simulation"]["volume_output"]
    w.output.volume_data.format = str(vol["format"])
    w.output.volume_data.fields = list(vol["fields"])
    w.output.volume_data.filename = str(vol["filename"])
    w.output.volume_data.sampling_interval_in_time_steps = int(vol["sampling_interval_steps"])

    # ---------------------------
    # Run
    # ---------------------------
    simcfg = cfg["simulation"]
    sn.api.run(
        input_file=w,
        site_name=str(simcfg["site_name"]),
        output_folder=str(data_out_dir),
        overwrite=True,
        delete_remote_files=True,
        wall_time_in_seconds=int(simcfg["wall_time_s"]),
        ranks=int(simcfg["ranks"]),
        get_all=True,
    )

    print(f"STF figure: {stf_png}")
    print(f"Simulation output folder: {data_out_dir}")


if __name__ == "__main__":
    main()
