export SINGULARITY_CACHEDIR=$SCRATCH
export SINGULARITY_TMPDIR=$SCRATCH

source .env

singularity build --fakeroot --bind ${PROJECT_ROOT}/DiffEqGMRFs.jl:/opt/DiffEqGMRFs.jl gmrf-pde.sif Singularity.def
