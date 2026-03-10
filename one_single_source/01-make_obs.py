#!/usr/bin/env python3
"""
01_make_obs.py

Generate a simple observation geometry (one source + a vertical receiver line),
write it to JSON, and save a quick-look plot.

Date: 10 Mar 2026
Author: Shi Yao (yaoshi229@gmail.com)

Inputs:
- config.yaml (must be in the same directory as this script)

Outputs:
- paths.obs_json: observation geometry in JSON
- paths.fig_dir/obs_system.png: geometry plot
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml


def main():
    # ---------------------------
    # Load config.yaml
    # ---------------------------
    cfg_path = Path(__file__).with_name("config.yaml")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    # ---------------------------
    # Paths from config
    # ---------------------------
    fig_dir = Path(cfg["paths"]["fig_dir"])
    obs_json = Path(cfg["paths"]["obs_json"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    obs_json.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Domain bounds from config
    # ---------------------------
    dom = cfg["domain"]
    xmin, xmax = float(dom["x0"]), float(dom["x1"])
    ymin, ymax = float(dom["y0"]), float(dom["y1"])

    # Absorbing boundary distance
    abc_dist = float(cfg["absorbing_boundary"]["width_m"])

    # ---------------------------
    # Source and receivers
    # ---------------------------
    src_x = np.array([6000.0])
    src_y = np.zeros_like(src_x)

    rec_y = np.arange(-301, 0, 20)
    rec_x = np.ones_like(rec_y) * 14000.0

    # ---------------------------
    # Write obs.json
    # ---------------------------
    obs_dict = {
        "sources": {
            f"src.{i}": {"x": float(src_x[i]), "y": float(src_y[i])}
            for i in range(len(src_x))
        },
        "receivers": {
            f"XX.{j}": {"x": float(rec_x[j]), "y": float(rec_y[j])}
            for j in range(len(rec_x))
        },
    }

    with obs_json.open("w") as f:
        json.dump(obs_dict, f, indent=4)

    # ---------------------------
    # Plot
    # ---------------------------
    rect_x_min, rect_x_max = xmin + abc_dist, xmax - abc_dist
    rect_y_min, rect_y_max = ymin + abc_dist, ymax

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(src_x, src_y, color="red", marker="o", label="Source")
    ax.scatter(rec_x, rec_y, color="blue", marker="v", label="Receivers")

    # Inner box (excluding absorbing boundary on x0/x1/y0)
    ax.plot([rect_x_min, rect_x_max], [rect_y_min, rect_y_min], color="red")
    ax.plot([rect_x_min, rect_x_min], [rect_y_min, rect_y_max], color="red")
    ax.plot([rect_x_max, rect_x_max], [rect_y_min, rect_y_max], color="red")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Observation System")
    ax.legend()
    ax.grid(True)

    out_fig = fig_dir / "obs_system.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {obs_json}")
    print(f"Wrote: {out_fig}")


if __name__ == "__main__":
    main()
