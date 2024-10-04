using DrWatson
@quickactivate "DiffEqGMRFs"

using Distributions,
    DiffEqGMRFs,
    GMRFs,
    Ferrite,
    HDF5,
    SparseArrays,
    LinearAlgebra,
    LinearMaps,
    Printf,
    TimerOutputs,
    Random,
    ArgParse,
    Logging,
    LoggingExtras
using Distributions
import Base: show

logger = FormatLogger() do io, args
    println(io, "[", args.level, "] ", args.message)
end;

global_logger(logger)

###### Argparse ######
function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--datasetname"
        help = "Name of the Darcy flow dataset to use"
        arg_type = String
        default = "piececonst_r241_N1024_smooth2"
        "--N_xy"
        help = "Number of FEM elements in each direction"
        arg_type = Int
        default = 300
        "--dry_run"
        help = "Test run which does not go through the entire dataset"
        arg_type = Bool
        default = true
        "--boundary_width"
        help = "Width of the inflated boundary"
        arg_type = Float64
        default = 0.0
    end
    return parse_args(s)
end
parsed_args = parse_cmd()

rng = MersenneTwister(1854390)
###### Read data ######
datasetname = parsed_args["datasetname"]
N_xy = parsed_args["N_xy"]
boundary_width = parsed_args["boundary_width"]
dry_run = parsed_args["dry_run"]
beta = 1.0
params = @strdict datasetname N_xy boundary_width dry_run beta

@info params

const to = TimerOutput()

###### Read data ######
path = datadir("input_data", "Darcy_241", "$datasetname.mat")
ds = DarcyDataset(path)
example_soln, example_coeff = get_problem(ds, 2)
x_coords, y_coords = ds.x_coords, ds.y_coords

inflated_boundary = boundary_width > 0.0

###### Form discretization ######
uniform_unit_square_discretization(N_xy; boundary_width=boundary_width, element_order=2, use_dirichlet_bc=!inflated_boundary) # Trigger precompilation
@timeit to "Mesh generation" disc = uniform_unit_square_discretization(N_xy; boundary_width=boundary_width, element_order=2, use_dirichlet_bc=!inflated_boundary)

@timeit to "Etc" begin
    pred_coords = [Tensors.Vec(Float64(x), Float64(y)) for x in x_coords for y in y_coords]
    E = evaluation_matrix(disc, pred_coords)
end

function to_mat(dof_vals, E, x_coords, y_coords)
    pred = E * dof_vals
    return reshape(pred, (length(x_coords), length(y_coords)))'
end

###### Prior ######
function form_prior(disc::FEMDiscretization, ν_matern = 2, range = 0.05, σ² = 1.0)
    κ = √(8ν_matern) / range
    spde = MaternSPDE{2}(κ, ν_matern, σ²)

    return GMRFs.discretize(spde, disc)
end

form_prior(disc, 2, 1 / sqrt(N_xy)) # Trigger precompilation
@timeit to "Prior construction" x = form_prior(disc, 2, 1 / sqrt(N_xy))

@timeit to "Etc" cbp = CholeskySolverBlueprint(RBMCStrategy(50, rng))

@timeit to "Set up collocation matrices" begin
    coll_step = (1 / (2 * N_xy))
    coll_range = coll_step:coll_step:(1 - coll_step)
    coll_grid = [Tensors.Vec(x, y) for x in coll_range for y in coll_range]
    ∂²u∂x², ∂²u∂y² =
        second_derivative_matrices(disc, coll_grid; derivative_idcs = [(1, 1), (2, 2)])
    D = -(∂²u∂x² + ∂²u∂y²)
    y = beta * ones(size(D, 1))
end


##### DARCY OBSERVATIONS ######
function form_observations(
    disc::FEMDiscretization,
    example_coeff;
    inflated_boundary = false,
    N_xy = 300,
)
    coeffs = map(x -> get_xy_idcs(x, x_coords, y_coords), coll_grid)
    coeffs = [example_coeff[point[1], point[2]] for point in coeffs]
    coeff_diag = Diagonal(coeffs)
    D_flow = (1e-5 * coeff_diag * D)

    A, ys = D_flow, (1e-5 * y)

    if inflated_boundary
        boundary_step = (1 / N_xy)
        boundary_obs = [
            [Tensors.Vec(x, 0.0) for x = 0:boundary_step:1.0]
            [Tensors.Vec(x, 1.0) for x = 0:boundary_step:1.0]
            [Tensors.Vec(0.0, y) for y = 0:boundary_step:1.0]
            [Tensors.Vec(1.0, y) for y = 0:boundary_step:1.0]
        ]
        A_boundary = evaluation_matrix(disc, boundary_obs)
        ys_boundary = zeros(size(A_boundary, 1))
        A = [A; A_boundary]
        ys = [ys; ys_boundary]
    end
    return A, ys
end


A, ys = form_observations(
    disc,
    example_coeff;
    inflated_boundary = inflated_boundary,
    N_xy = N_xy
)
Q_ϵ = 1e8

condition_on_observations(x, A, Q_ϵ, ys; solver_blueprint = cbp) # Trigger precompilation
@timeit to "Conditioning + Node reordering" x_cond =
    condition_on_observations(x, A, Q_ϵ, ys; solver_blueprint = cbp) # For Cholesky permutation
mat_nnz = nnz(sparse(to_matrix(precision_map(x_cond))))
if x_cond isa ConstrainedGMRF
    p = x_cond.inner_gmrf.solver_ref[].precision_chol.p
    chol_nnz = nnz(x_cond.inner_gmrf.solver_ref[].precision_chol)
else
    p = x_cond.solver_ref[].precision_chol.p
    chol_nnz = nnz(x_cond.solver_ref[].precision_chol)
end

cbp2 = CholeskySolverBlueprint(RBMCStrategy(50, rng), p)

function solve_problem(idx)
    cur_to = TimerOutput()
    example_soln, example_coeff = get_problem(ds, idx)
    @timeit cur_to "PDE Discretization" A, ys = form_observations(
        disc,
        example_coeff;
        inflated_boundary = inflated_boundary,
        N_xy = N_xy
    )
    @timeit cur_to "Conditioning" x_cond =
        condition_on_observations(x, A, Q_ϵ, ys; solver_blueprint = cbp2)
    pred = to_mat(full_mean(x_cond), E, x_coords, y_coords)
    @timeit cur_to "Sampling" full_rand(rng, x_cond)
    @timeit cur_to "Std dev" cur_std = std(x_cond)
    cur_rel_err = rel_err(pred, example_soln)
    cur_rmse = rmse(pred, example_soln)
    cur_max_err = max_err(pred, example_soln)
    std_norm = norm(cur_std)
    return cur_rel_err, cur_rmse, cur_max_err, std_norm, cur_to
end

rel_errs = Float64[]
rmses = Float64[]
max_errs = Float64[]
std_norms = Float64[]
conditioning_times = Int64[]
std_dev_times = Int64[]
sampling_times = Int64[]
pde_disc_times = Int64[]

N_samples = dry_run ? 3 : Base.size(ds.darcy_vars["sol"], 1)
for i = 1:N_samples
    cur_rel_err, cur_rmse, cur_max_err, std_norm, cur_to = solve_problem(i)
    push!(rel_errs, cur_rel_err)
    push!(rmses, cur_rmse)
    push!(max_errs, cur_max_err)
    push!(std_norms, std_norm)
    push!(conditioning_times, TimerOutputs.time(cur_to["Conditioning"]))
    push!(std_dev_times, TimerOutputs.time(cur_to["Std dev"]))
    push!(sampling_times, TimerOutputs.time(cur_to["Sampling"]))
    push!(pde_disc_times, TimerOutputs.time(cur_to["PDE Discretization"]))
    if i % 10 == 0
        @info "Finished $i / $N_samples ($((i / N_samples) * 100)%)"
    end
end

out_dict = @strdict rel_errs rmses max_errs std_norms conditioning_times std_dev_times sampling_times pde_disc_times
out_dict = merge(out_dict, params, @strdict mat_nnz chol_nnz)

@tagsave(datadir("sims", "darcy", "gmrf-colloc", savename(params, "jld2")), out_dict)
