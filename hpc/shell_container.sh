source .env

singularity shell --bind ${PROJECT_ROOT}/DiffEqGMRFs.jl:/opt/DiffEqGMRFs.jl,${PROJECT_ROOT}/GMRFs.jl:/opt/GMRFs.jl gmrf-pde.sif