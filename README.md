# A Numerical Framework for Newtonian-Noise Estimation

This repository contains two Salvus-based 2D spectral-element examples for estimating seismic Newtonian noise in underground gravitational-wave detector settings. The workflows cover the full chain from geometry generation and wavefield simulation to displacement decomposition, Newtonian-noise force evaluation, and plotting.

The repository currently includes two worked examples:

- `one_single_source/`: one vertical surface-force source driven by a Ricker wavelet.
- `an_array_30sources/`: 30 randomly distributed surface sources driven by band-limited random source-time functions.

## Repository Layout

Each example folder is self-contained and follows the same high-level structure:

- `config.yaml`: model, simulation, decomposition, and Newtonian-noise parameters.
- numbered Python scripts: sequential workflow steps.
- `data/`: input metadata such as `obs.json` and, for the 30-source example, generated STFs.
- `figures/`: geometry, STF, wavefield, and Newtonian-noise plots.
- `output/`: generated mesh, simulation output, derived wavefields, and final result arrays.

## Requirements

The scripts rely on a local Python environment with:

- `numpy`
- `scipy`
- `matplotlib`
- `PyYAML`
- `h5py`
- `xarray`
- `h5netcdf`
- Salvus Python packages such as `salvus.namespace`, `salvus.mesh`, and `salvus.toolbox`

Salvus is not vendored in this repository, so you will need access to an existing Salvus installation and a configured site name that matches the `simulation.site_name` entry in each `config.yaml`.

## Workflow Overview

Both examples follow the same scientific workflow:

1. Define source and receiver geometry.
2. Build a 2D mesh in Salvus.
3. Run a wave propagation simulation.
4. Convert the wavefield into displacement, divergence, and curl products.
5. Compute Newtonian-noise force contributions at the test-mass location.
6. Plot and inspect the results.

The `an_array_30sources/` example adds an STF-generation step and an optional P-wave fraction analysis.

## Example 1: Single Source

Working directory:

```bash
cd one_single_source
```

Run the full workflow in order:

```bash
python 01-make_obs.py
python 02-build_mesh.py
python 03-run_simulation.py
python 04-decompose.py
python 05-calc_NN.py
python 06-eval_NN.py
```

What each step does:

- `01-make_obs.py`: creates `data/obs.json` and `figures/obs_system.png`.
- `02-build_mesh.py`: builds the Salvus HDF5 mesh defined in `config.yaml`.
- `03-run_simulation.py`: runs the waveform simulation and writes `output/sim/volume_output.h5`.
- `04-decompose.py`: computes displacement, divergence, and curl products in `output/derived/`.
- `05-calc_NN.py`: evaluates bulk and dipole Newtonian-noise forces and stores arrays in `output/results/`.
- `06-eval_NN.py`: generates Newtonian-noise plots in `figures/NN_results/`.

## Example 2: Thirty Random Sources

Working directory:

```bash
cd an_array_30sources
```

Run the full workflow in order:

```bash
python 01-1_make_obs.py
python 01-2_gen_stfs.py
python 02-build_mesh.py
python 03_simulation.py
python 04_decompose.py
python 05-calc_NN.py
python 06-eval_NN.py
python 07-calc_p.py
```

What each step does:

- `01-1_make_obs.py`: generates the source and receiver geometry.
- `01-2_gen_stfs.py`: creates random band-limited STFs and an example STF plot.
- `02-build_mesh.py`: builds the Salvus mesh.
- `03_simulation.py`: runs the multi-source waveform simulation.
- `04_decompose.py`: samples the wavefield on a regular grid and saves displacement, divergence, and curl.
- `05-calc_NN.py`: computes bulk and dipole Newtonian-noise force time series.
- `06-eval_NN.py`: plots force traces and Newtonian-noise strain ASD estimates.
- `07-calc_p.py`: estimates the local P-wave fraction in a user-defined window.

## Configuration Notes

Most user-facing parameters live in `config.yaml` inside each example directory, including:

- model domain and absorbing boundaries
- mesh resolution
- simulation time step and run duration
- source properties
- decomposition grid and time window
- test-mass position, orientation, and Newtonian-noise integration settings

For the 30-source example, `config.yaml` also controls source placement, STF generation, and the analysis window used for the P-wave fraction estimate.

## Outputs

Typical generated outputs include:

- geometry and STF figures under `figures/`
- wavefield output under `output/sim/`
- decomposed displacement, divergence, and curl fields under `output/derived/`
- Newtonian-noise arrays under `output/results/<project><tag>/`
- final plots such as `NN_forces.svg`, `NN_strain_ASD.svg`, and `p_*.svg`

Some Salvus-generated HDF5 files can become very large. To keep the repository lightweight and GitHub-friendly, avoid committing generated artifacts near or above 100 MB. The small example outputs already tracked here are useful for inspection, but large local simulation products should stay out of git history.

## Notes

- The repository does not currently ship with a pinned environment file.
- The scripts assume they are executed from within their respective example directories.
- An associated paper is mentioned in the original project summary, but no publication link is currently included in the repository.

## License

This project is distributed under the terms of the [LICENSE](LICENSE) file.
