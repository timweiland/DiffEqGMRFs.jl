# DiffEqGMRFs

[![Build Status](https://github.com/timweiland/DiffEqGMRFs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/timweiland/DiffEqGMRFs.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/timweiland/DiffEqGMRFs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/timweiland/DiffEqGMRFs.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This repo contains the experiments for our paper "*Flexible and Efficient Probabilistic PDE Solvers through Gaussian Markov Random Fields*".

## Installation

1. [Download Julia (>= version 1.10)](https://julialang.org/downloads/).

2. Clone this repo.

3. In the root directory of this project, run `julia --project=.` followed by `] dev .`

Please note that GaussianMarkovRandomFields.jl (the main dependency for these experiments) is not registered yet.
Until it is, you can install it directly from GitHub through
`] add https://github.com/timweiland/GaussianMarkovRandomFields.jl`.

## Running the experiments
`./hpc` contains utilities to get this code running in a singularity container for HPC.

Once you've figured this out, you can grab the data from the sources specified in the subdirectories of `./data`.

Afterwards, you can run the scripts in `./scripts`.
