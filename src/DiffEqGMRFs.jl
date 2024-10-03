module DiffEqGMRFs

# Write your package code here.
include("datasets/darcy.jl")
include("spdes/shallow_water.jl")
include("utils.jl")
include("metrics.jl")
include("tridiagonal_cholesky.jl")

end
