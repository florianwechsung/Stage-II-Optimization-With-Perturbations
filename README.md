# A-posteriori optimization for increasing manufacturing tolerances in stellarator coil design

This repository contains code to accompany the manuscript


- *A-posteriori optimization for increasing manufacturing tolerances in stellarator coil design*, Florian Wechsung, Andrew Giuliani, Matt Landreman, Antoine Cerfon, Georg Stadler

## Context

In 

- *Magnetic fields with precise quasisymmetry for plasma confinement.* Landreman, Matt, and Elizabeth Paul. Physical Review Letters 128.3 (2022): 035001, [10.1103/PhysRevLett.128.035001](https://doi.org/10.1103/PhysRevLett.128.035001).

magnetic fields were discovered that satisfy precise quasi-symmetry, resulting in extremely good confinement properties. Later, in

- *Precise stellarator quasi-symmetry can be achieved with electromagnetic coils*, Florian Wechsung, Matt Landreman , Andrew Giuliani, Antoine Cerfon, and Georg Stadler, to appear in Proceeding of the National Academy of Sciences

coils were found that are able to reproduce these magnetic fields very well.

The goal of this work is now to understand and mitigate the impact of coil manufacturing errors on the performance of the magnetic field.



## Installation

To use this code, first clone the repository including all its submodules, via

    git clone --recursive 

and then install [SIMSOPT](https://github.com/hiddenSymmetries/simsopt) via

    pip install -e simsopt/

If you have any trouble with the installation of SIMSOPT, please refer to the installation instructions [here](https://simsopt.readthedocs.io/en/latest/installation.html#virtual-environments) or open an [issue](https://github.com/hiddenSymmetries/simsopt/issues).

## Basic Usage

Once you have installed SIMSOPT, you can run

    python3 driver.py --order 16 --lengthbound 18 --mindist 0.10 --maxkappa 5 --maxmsc 5  --ig 0 --expquad

to find coils that approximate the **QA** configuration in Landreman \& Paul. The configuration you obtain from this command corresponds to the `QA[18]` configuration in the Precise QS paper.

Here the options mean:

- `order`: at what order to truncate the Fourier series for the curves that represent the coils.
- `lengthbound`: the maximum total length of the four modular coils.
- `mindist`: the minimum distance between coils to be enforced.
- `maxkappa`: the maximum allowable curvature of the coils.
- `maxmsc`: the maximum allowable mean squared curvature of the coils.
- `ig`: which initial guess to choose. `0` corresponds to flat coils, other values result in random perturbations of the flat coils. In the paper we picked `ig\in\{0,...,7\}`
- `expquad`: turns on a modified quadrature scheme that results in exponential
  convergence of the surface integral in the objective(recommended).

If you would like to target the QA+Well configuration instead, simply add `--well` to the command.

If you want to run the stochastic optimization with 4096 samples using MPI:

    python3 driver.py --order 16 --lengthbound 18 --mindist 0.10 --maxkappa 5 --maxmsc 5  --ig 0 --expquad --nsamples 4096 --sigma 1e-3

This is an expensive simulation, so you either want to do this on a cluster with MPI or reduce the number of samples.
You can use the result of a previous deterministic optimization by appending `--usedetig` to the command.

## Analysis

Since the optimization routine may find a different minimizer depending on the initial guess, we first run

    python3 eval_find_best.py  --order 16 --lengthbound 18 --mindist 0.10 --maxkappa 5 --maxmsc 5 --expquad
    python3 eval_find_best.py  --order 16 --lengthbound 18 --mindist 0.10 --maxkappa 5 --maxmsc 5 --expquad --nsamples 4096 --sigma 1e-3
    python3 eval_find_best.py  --order 16 --lengthbound 18 --mindist 0.10 --maxkappa 5 --maxmsc 5 --expquad --nsamples 4096 --sigma 1e-3 --usedetig

to find the best of the local minimizers. Once this is done, we can compute basic properties of the coils and the magnetic field.

First run

    python3 eval_geo.py

to print geometric properties of the coils. Then run

    python3 eval_vis.py

to create some paraview files that visualize both the coils and field.

We provide slurm scripts for all expensive optimizations and evaluations

- `job_det.slurm`: to run the deterministic optimization.
- `job_arg.slurm`: to be used with `submit_job_arg.sh` to run the stochastic optimization runs.
- `job_corrections.slurm`: to compute the coil placement and current corrections.
- `job_bmn.slurm`: to compute the deviation from quasi-symmetry.
- `job_qfm.slurm`: to compute qfm surface used for the spawning of the particles and as the boundary when particles are considered lost. You need to run `eval_compute_multiple_qfm.py` first.
- `job_particles_det.slurm`: to compute particle confinement numbers for the exactly build configurations.
- `job_particles_samples.slurm`: to compute particle confinement numbers for perturbed coils.
