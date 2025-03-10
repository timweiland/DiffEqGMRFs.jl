source .env

singularity exec --bind ${PROJECT_ROOT}/DiffEqGMRFs.jl:/opt/DiffEqGMRFs.jl gmrf-pde.sif "$@"
