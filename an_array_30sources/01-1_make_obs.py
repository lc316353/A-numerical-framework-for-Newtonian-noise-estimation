#!/usr/bin/env python3
"""
01-1_make_obs.py

Generate obs.json for multiple random sources.

Inputs
- config.yaml 

Outputs
- paths.obs_json
- paths.fig_dir/obs_system.png

Date: 10 Mar 2026
Author: Shi Yao (yaoshi229@gmail.com)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml


def main():
    cfg_path = Path(__file__).with_name("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find {cfg_path}. Put config.yaml in the same directory as this script.")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    # Output locations
    fig_dir = Path(cfg["paths"]["fig_dir"])
    obs_json = Path(cfg["paths"]["obs_json"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    obs_json.parent.mkdir(parents=True, exist_ok=True)

    # Domain + ABC 
    dom = cfg["domain"]
    xmin, xmax = float(dom["x0"]), float(dom["x1"])
    ymin, ymax = float(dom["y0"]), float(dom["y1"])
    abc_dist = float(cfg["absorbing_boundary"]["width_m"])

    # Source parameters
    nsources = int(cfg["sources"]["nsources"])
    seed = int(cfg["sources"]["seed"])
    loc = cfg["sources"]["locations"]

    if nsources % 2 != 0:
        raise ValueError("sources.nsources must be even to split left/right bands (e.g., 30).")

    n_half = nsources // 2
    rng = np.random.default_rng(seed)

    x_left = rng.random(n_half) * float(loc["x_left_width"]) + float(loc["x_left_min"])
    x_right = rng.random(n_half) * float(loc["x_right_width"]) + float(loc["x_right_min"])
    src_x = np.concatenate([x_left, x_right])
    src_y = np.ones_like(src_x) * float(loc["y_source"])

    # Receivers
    rec_y = np.arange(-20.0, 1.0, 20.0)
    rec_x = np.ones_like(rec_y) * 10000.0

    # Write obs.json
    obs_dict = {
        "sources": {f"src.{i}": {"x": float(src_x[i]), "y": float(src_y[i])} for i in range(len(src_x))},
        "receivers": {f"XX.{j}": {"x": float(rec_x[j]), "y": float(rec_y[j])} for j in range(len(rec_x))},
    }
    with obs_json.open("w") as f:
        json.dump(obs_dict, f, indent=4)

    # Plot geometry
    rect_x_min, rect_x_max = xmin + abc_dist, xmax - abc_dist
    rect_y_min, rect_y_max = ymin + abc_dist, ymax

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(src_x, src_y, color="red", marker="o", s=20, label="Sources")
    ax.scatter(rec_x, rec_y, color="blue", marker="v", s=12, label="Receivers")

    ax.plot([rect_x_min, rect_x_max], [rect_y_min, rect_y_min], color="red")
    ax.plot([rect_x_min, rect_x_min], [rect_y_min, rect_y_max], color="red")
    ax.plot([rect_x_max, rect_x_max], [rect_y_min, rect_y_max], color="red")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Observation System ({nsources} sources)")
    ax.legend()
    ax.grid(True)

    out_fig = fig_dir / "obs_system.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {obs_json}")
    print(f"Wrote: {out_fig}")
    print(f"Sources: {len(src_x)} | Receivers: {len(rec_x)}")


if __name__ == "__main__":
    main()
