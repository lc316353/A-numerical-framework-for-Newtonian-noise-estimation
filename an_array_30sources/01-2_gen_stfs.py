#!/usr/bin/env python3
"""
01-2_gen_stfs_taper.py

Generate random noise source-time-functions (STFs) in frequency domain with
cosine tapers and random phase, then write them to HDF5.

Date: 10 Mar 2026
Author: Shi Yao (yaoshi229@gmail.com)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py
import yaml


def main():
    cfg_path = Path(__file__).with_name("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find {cfg_path}. Put config.yaml in the same directory as this script.")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    scfg = cfg["sources"]
   

    nsources = int(scfg["nsources"])
    seed = int(scfg["seed"])
    duration = float(scfg['stf']["duration_s"])
    fs = float(scfg['stf']["fs_hz"])
    dt = 1.0 / fs
    npts = int(round(fs * duration))

    fmin = float(scfg['stf']["fmin_hz"])
    fmax = float(scfg['stf']["fmax_hz"])
    f_taper_start = float(scfg['stf']["f_taper_start_hz"])
    amp_scale = float(scfg['stf']["amplitude"])

    output_file = Path(scfg['stf']["output_h5"])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plot_example = bool(scfg['stf'].get("plot_example", True))
    plot_file = Path(scfg['stf'].get("plot_file", "figures/stf_1_noise_window.png"))
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generate {nsources} random STFs | duration={duration}s | fs={fs} Hz | npts={npts}")
    print(f"Band: {fmin}-{fmax} Hz | high taper starts at {f_taper_start} Hz | seed={seed}")
    print(f"Output: {output_file}")

    freqs = np.fft.rfftfreq(npts, d=dt)
    rng = np.random.default_rng(seed)

    with h5py.File(output_file, "w") as h5file:
        h5file.attrs["seed"] = seed
        h5file.attrs["duration_s"] = duration
        h5file.attrs["fs_hz"] = fs
        h5file.attrs["fmin_hz"] = fmin
        h5file.attrs["fmax_hz"] = fmax
        h5file.attrs["f_taper_start_hz"] = f_taper_start
        h5file.attrs["amplitude"] = amp_scale

        for i in range(nsources):
            # Random phase
            random_phases = np.exp(1j * 2 * np.pi * rng.random(freqs.shape))

            # Amplitude spectrum: cosine taper low + flat + cosine taper high
            amplitude = np.zeros_like(freqs)

            low = (freqs >= 0) & (freqs < fmin)
            amplitude[low] = 0.5 * (1 - np.cos(np.pi * freqs[low] / fmin))

            flat = (freqs >= fmin) & (freqs <= f_taper_start)
            amplitude[flat] = 1.0

            high = (freqs > f_taper_start) & (freqs <= fmax)
            amplitude[high] = 0.5 * (
                1 + np.cos(np.pi * (freqs[high] - f_taper_start) / (fmax - f_taper_start))
            )

            spectrum = amplitude * random_phases
            time_signal = np.fft.irfft(spectrum, n=npts)

            # Normalize then scale
            m = np.max(np.abs(time_signal))
            if m > 0:
                time_signal = (time_signal / m) * amp_scale

            h5file.create_dataset(f"stf_{i+1}", data=time_signal)

    # Plot example STF #1
    if plot_example:
        with h5py.File(output_file, "r") as h5file:
            time_signal = h5file["stf_1"][:]

        spectrum = np.fft.rfft(time_signal, n=npts)
        amp_spec = np.abs(spectrum)
        t = np.arange(npts) / fs

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        axs[0].plot(freqs, amp_spec)
        axs[0].set_title("Amplitude Spectrum of stf_1")
        axs[0].set_xlabel("Frequency (Hz)")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_xlim(0, max(12, fmax + 1))
        axs[0].grid(True)

        axs[1].plot(t, time_signal)
        axs[1].set_title("Time-Domain Signal of stf_1")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Amplitude")
        axs[1].grid(True)

        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved example plot: {plot_file}")


if __name__ == "__main__":
    main()
