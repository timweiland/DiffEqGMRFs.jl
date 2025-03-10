source .env

singularity shell --bind ${PROJECT_ROOT}/DiffEqGMRFs.jl:/opt/DiffEqGMRFs.jl gmrf-pde.sif
