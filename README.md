# A Numerical Framework for Newtonian-Noise Estimation

Newtonian noise is expected to limit the sensitivity of ground-based gravitational-wave detectors at low frequencies. This repository presents code for a numerical framework for Newtonian-noise estimation based on spectral-element simulations of a seismic wave field with Salvus.

The repository currently includes two worked examples:

- `one_single_source/`: one vertical surface-force source driven by a Ricker wavelet.
- `an_array_30sources/`: 30 randomly distributed surface sources driven by band-limited random source-time functions.

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

Salvus is not vendored in this repository, so you will need access to an existing Salvus installation and license.

## Workflow Overview

Both examples follow the same high-level workflow:

1. Define source and receiver geometry.
2. Build a 2D mesh in Salvus.
3. Run a wave propagation simulation.
4. Decompose the wavefield into displacement, divergence, and curl products.
5. Compute Newtonian-noise force contributions at the test-mass location.
6. Plot and inspect the results.

## Associated paper

Schillings, P., Yao, S., Erdmann, J., & Rietbrock, A. (2026). *A numerical framework for Newtonian-noise estimation at the Einstein Telescope: 2-D simulations beyond the plane-wave approximation*. [arXiv:2603.15424](https://arxiv.org/abs/2603.15424).
