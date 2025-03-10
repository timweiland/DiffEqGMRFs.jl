source .env

singularity exec --bind ${PROJECT_ROOT}/DiffEqGMRFs.jl:/opt/DiffEqGMRFs.jl,${PROJECT_ROOT}/GaussianMarkovRandomFields.jl:/opt/GaussianMarkovRandomFields.jl gmrf-pde.sif "$@"