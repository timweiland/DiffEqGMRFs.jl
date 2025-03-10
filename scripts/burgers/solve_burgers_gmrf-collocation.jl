using DrWatson
@quickactivate "DiffEqGMRFs"

using Distributions,
    DiffEqGMRFs,
    GaussianMarkovRandomFields,
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
    LoggingExtras,
    MAT
using Distributions
import Base: show

logger = FormatLogger() do io, args
    if args.level == Logging.Debug
        return
    end
    println(io, "[", args.level, "] ", args.message)
end;

global_logger(logger)

###### Argparse ######
function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--datasetname"
        help = "Name of the Burgers dataset to use"
        arg_type = String
        default = "burgers_v100_t100_r1024_N2048"
        "--N_basis"
        help = "Number of FEM basis elements"
        arg_type = Int
        default = 750
        "--N_collocation"
        help = "Number of collocation points"
        arg_type = Int
        default = 750
        "--prior_type"
        help = "Prior type to use. adv_diff or product_matern"
        arg_type = String
        default = "adv_diff"
        "--matern_temporal_lengthscale"
        help = "Temporal lengthscale for Matern prior"
        arg_type = Float64
        default = 3.0
        "--matern_spatial_lengthscale"
        help = "Spatial lengthscale for Matern prior"
        arg_type = Float64
        default = 0.02
        "--dry_run"
        help = "Test run which does not go through the entire dataset"
        arg_type = Bool
        default = true
        "--N_samples"
        help = "Number of example problems to solve"
        arg_type = Int
        default = 30
    end
    return parse_args(s)
end
parsed_args = parse_cmd()

rng = MersenneTwister(985368934)
###### Read data ######
datasetname = parsed_args["datasetname"]
N_basis = parsed_args["N_basis"]
N_collocation = parsed_args["N_collocation"]
prior_type = parsed_args["prior_type"]
matern_temporal_lengthscale = parsed_args["matern_temporal_lengthscale"]
matern_spatial_lengthscale = parsed_args["matern_spatial_lengthscale"]
dry_run = parsed_args["dry_run"]
N_samples = parsed_args["N_samples"]

parameters = @strdict datasetname N_basis N_collocation prior_type matern_temporal_lengthscale matern_spatial_lengthscale dry_run N_samples

@info parameters

const to = TimerOutput()

###### Read data ######
path = datadir("input_data", "Burgers", "$datasetname.mat")
ds = BurgersDataset(path)
example_ic, example_solution = get_initial_condition(ds, 2), get_solution(ds, 2)
x_coords, ts = ds.x_coords, ds.ts

###### Form discretization ######
periodic_unit_interval_discretization(N_basis; element_order=2) # Trigger precompilation
@timeit to "Mesh generation" disc = periodic_unit_interval_discretization(N_basis; element_order=2)

@timeit to "Etc" begin
    pred_coords = [Tensors.Vec(Float64(x)) for x in x_coords]
    E = evaluation_matrix(disc, pred_coords)
    E = vcat([spatial_to_spatiotemporal(E, t, length(ts)) for t in eachindex(ts)]...)
end

function to_mat(dof_vals, E, ts, x_coords)
    pred = E * dof_vals
    return reshape(pred, (length(x_coords), length(ts)))'
end

###### Prior ######
function form_adv_diff_prior(disc::FEMDiscretization, ts, ic, ν_burgers, matern_spatial_lengthscale)
    bulk_speed = mean(ic)

    ν_matern = 3 // 2
    desired_range = matern_spatial_lengthscale
    κ = √(8ν_matern) / desired_range

    c = 1 / (ν_burgers)
    γ = -c * bulk_speed
    spde = AdvectionDiffusionSPDE{1}(
        κ=0.0,
        α=1 // 1,
        H=1.0 * ones(1, 1),
        γ=[γ],
        c=c,
        τ=0.1 * sqrt(c),
        spatial_spde = MaternSPDE{1}(κ=κ, ν=ν_matern),
        initial_spde = MaternSPDE{1}(κ=κ, ν=ν_matern)
    )

    return GaussianMarkovRandomFields.discretize(spde, disc, ts; mean_offset = bulk_speed, prescribed_noise = 1e-8)
end

function form_product_matern_prior(disc::FEMDiscretization, ts, matern_temporal_lengthscale, matern_spatial_lengthscale)
    temporal_matern = MaternSPDE{1}(range=matern_temporal_lengthscale, smoothness=0, σ²=0.1)
    spatial_matern = MaternSPDE{1}(range=matern_spatial_lengthscale, smoothness=3, σ²=0.1)

    return product_matern(temporal_matern, length(ts), spatial_matern, disc; solver_blueprint = CholeskySolverBlueprint(var_strategy=RBMCStrategy(50)))
end

function form_prior(disc::FEMDiscretization, ts, ic, ν_burgers, prior_type, matern_temporal_lengthscale, matern_spatial_lengthscale)
    if prior_type == "adv_diff"
        return form_adv_diff_prior(disc, ts, ic, ν_burgers, matern_spatial_lengthscale)
    elseif prior_type == "product_matern"
        return form_product_matern_prior(disc, ts, matern_temporal_lengthscale, matern_spatial_lengthscale)
    else
        error("Unknown prior type: $prior_type")
    end
end

x = form_prior(disc, ts, example_ic, ds.ν, prior_type, matern_temporal_lengthscale, matern_spatial_lengthscale)
@timeit to "Prior construction" x = form_prior(disc, ts, example_ic, ds.ν, prior_type, matern_temporal_lengthscale, matern_spatial_lengthscale)

@timeit to "Etc" cbp = CholeskySolverBlueprint(var_strategy=RBMCStrategy(50; rng=rng))

A_ic = evaluation_matrix(disc, [Tensors.Vec(Float64(x)) for x in x_coords])
A_ic = spatial_to_spatiotemporal(A_ic, 1, length(ts))
ys_ic = example_ic

x_ic = condition_on_observations(x, A_ic, 1e8, ys_ic; solver_blueprint = cbp)

### Collocation
@timeit to "PDE Discretization (Linear part)" begin
    dx = 1 / N_collocation
    coll_grid = range(x_coords[1] + dx, x_coords[end] - dx, length = N_collocation)
    coll_grid = [Tensors.Vec(x) for x in coll_grid]
    A_coll = evaluation_matrix(disc, coll_grid)
    ∂u∂x = derivative_matrices(disc, coll_grid; derivative_idcs = [1])[1]
    ∂²u∂x² = second_derivative_matrices(disc, coll_grid; derivative_idcs = [(1, 1)])[1]

    dt = Float64(step(ts))
    Aₜ = vcat([spatial_to_spatiotemporal(A_coll, i, length(ts)) for i = 1:(length(ts)-1)]...)
    Aₜ₊₁ = vcat([spatial_to_spatiotemporal(A_coll, i, length(ts)) for i = 2:length(ts)]...)
    ∂uₜ₊₁∂x = vcat([spatial_to_spatiotemporal(∂u∂x, i, length(ts)) for i = 2:length(ts)]...)
    ∂²uₜ₊₁∂x² = vcat([spatial_to_spatiotemporal(∂²u∂x², i, length(ts)) for i = 2:length(ts)]...)
    y = spzeros(size(Aₜ, 1))
    #Aₜ, y = constrainify_linear_system(Aₜ, y, x_ic)
    #Aₜ₊₁, y = constrainify_linear_system(Aₜ₊₁, y, x_ic)
    #∂uₜ₊₁∂x, y = constrainify_linear_system(∂uₜ₊₁∂x, y, x_ic)
    #∂²uₜ₊₁∂x², y = constrainify_linear_system(∂²uₜ₊₁∂x², y, x_ic)

    ν_burger = ds.ν
    J_static = Aₜ₊₁ - Aₜ - dt * ν_burger * ∂²uₜ₊₁∂x²
end

function f_and_J(w)
    Aₜ₊₁_mul_w = Aₜ₊₁ * w
    ∂uₜ₊₁∂x_mul_w = ∂uₜ₊₁∂x * w
    f = Aₜ₊₁_mul_w - Aₜ * w + dt * Aₜ₊₁_mul_w .* ∂uₜ₊₁∂x_mul_w - dt * ν_burger * ∂²uₜ₊₁∂x² * w
    J = J_static + dt * (Diagonal(∂uₜ₊₁∂x_mul_w) * Aₜ₊₁ + Diagonal(Aₜ₊₁_mul_w) * ∂uₜ₊₁∂x)
    return f, J
end

noise_ic = 1e8
noise_collocation = 1e8

A_soln = evaluation_matrix(disc, [Tensors.Vec(Float64(x)) for x in x_coords])
A_soln = vcat([spatial_to_spatiotemporal(A_soln, t, length(ts)) for t in eachindex(ts)]...)

function interpolate_solution(x_prior, solution_mat, ys_ic)
    soln_mat = copy(solution_mat)
    soln_mat[1, :] = ys_ic
    ys_soln = reshape(soln_mat', (length(x_coords) * length(ts),))
    x_soln = condition_on_observations(x_prior, A_soln, 1e12, ys_soln; solver_blueprint = cbp)
    return mean(x_soln)
end

function get_log_det_Σ(x_final)
    L_diag = diag(sparse(x_final.solver_ref[].precision_chol.L))
    return -2 * sum(log.(L_diag))
end

function nll_soln(x_final, sqmahal_to_soln)
    return 0.5 * (length(x_final) * log(2π) + sqmahal_to_soln + get_log_det_Σ(x_final))
end

function solve_problem(idx)
    cur_to = TimerOutput()
    example_ic, example_soln = get_initial_condition(ds, idx), get_solution(ds, idx)
    example_soln_no_t0 = example_soln[2:end, 1:end]
    ys_ic = example_ic

    @timeit cur_to "Prior" x = form_prior(disc, ts, example_ic, ds.ν, prior_type, matern_temporal_lengthscale, matern_spatial_lengthscale)
    @timeit cur_to "Initial condition" x_ic = condition_on_observations(x, A_ic, noise_ic, ys_ic; solver_blueprint = cbp)

    ic_pred = to_mat(mean(x_ic), E, ts, x_coords)
    ic_pred = ic_pred[2:end, 1:end]
    ic_rel_err = rel_err(ic_pred, example_soln_no_t0)
    ic_rmse = rmse(ic_pred, example_soln_no_t0)
    ic_max_err = max_err(ic_pred, example_soln_no_t0)

    p = x_ic.solver_ref[].precision_chol.p
    gncbp = GNCholeskySolverBlueprint(p)


    gno = GaussNewtonOptimizer(
        mean(x_ic),
        precision_map(x_ic),
        f_and_J,
        noise_collocation,
        zeros(size(J_static, 1)),
        mean(x_ic);
        solver_bp=gncbp,
    )
    @timeit cur_to "Optimization" begin
        optimize(gno)

        J_final = gno.Jₖ
        Q = gno.Q_mat
        new_precision = LinearMap(Q + noise_ic * J_final' * J_final)
        x_final = ConcreteConstantMeshSTGMRF(
            gno.xₖ,
            new_precision,
            disc,
            CholeskySolverBlueprint(var_strategy=RBMCStrategy(50), perm=p),
        )
    end
    mat_nnz = nnz(to_matrix(precision_map(x_final)))
    chol_nnz = nnz(x_final.solver_ref[].precision_chol)

    soln_dofs = interpolate_solution(x, example_soln, ys_ic)
    sqmahal_to_soln = sqmahal(x_final, soln_dofs)
    cur_nll = nll_soln(x_final, sqmahal_to_soln)
    
    pred = to_mat(mean(x_final), E, ts, x_coords)
    pred = pred[2:end, 1:end]
    @timeit cur_to "Sampling" rand(rng, x_final)
    @timeit cur_to "Std dev" cur_std = std(x_final)
    cur_rel_err = rel_err(pred, example_soln_no_t0)
    cur_rmse = rmse(pred, example_soln_no_t0)
    cur_max_err = max_err(pred, example_soln_no_t0)
    std_norm = norm(cur_std)
    N_newton_steps = length(gno.r_obs_norm_history) - 1
    return cur_rel_err, cur_rmse, cur_max_err, ic_rel_err, ic_rmse, ic_max_err, std_norm, N_newton_steps, mat_nnz, chol_nnz, sqmahal_to_soln, cur_nll, cur_to
end

rel_errs = Float64[]
rmses = Float64[]
max_errs = Float64[]
ic_rel_errs = Float64[]
ic_rmses = Float64[]
ic_max_errs = Float64[]
std_norms = Float64[]
N_newton_steps = Int64[]
prior_times = Int64[]
initial_condition_times = Int64[]
std_dev_times = Int64[]
optimization_times = Int64[]
sampling_times = Int64[]
mat_nnzs = Int64[]
chol_nnzs = Int64[]
sqmahals = Float64[]
nlls = Float64[]

if dry_run
    N_samples = 3
end
@info "Beginning to solve $N_samples problems"
for i = 1:N_samples
    cur_rel_err, cur_rmse, cur_max_err, ic_rel_err, ic_rmse, ic_max_err, std_norm, N_newton_step, mat_nnz, chol_nnz, cur_sqmahal, cur_nll, cur_to = solve_problem(i)
    push!(rel_errs, cur_rel_err)
    push!(rmses, cur_rmse)
    push!(max_errs, cur_max_err)
    push!(ic_rel_errs, ic_rel_err)
    push!(ic_rmses, ic_rmse)
    push!(ic_max_errs, ic_max_err)
    push!(std_norms, std_norm)
    push!(N_newton_steps, N_newton_step)
    push!(mat_nnzs, mat_nnz)
    push!(chol_nnzs, chol_nnz)
    push!(sqmahals, cur_sqmahal)
    push!(nlls, cur_nll)
    push!(prior_times, TimerOutputs.time(cur_to["Prior"]))
    push!(initial_condition_times, TimerOutputs.time(cur_to["Initial condition"]))
    push!(std_dev_times, TimerOutputs.time(cur_to["Std dev"]))
    push!(optimization_times, TimerOutputs.time(cur_to["Optimization"]))
    push!(sampling_times, TimerOutputs.time(cur_to["Sampling"]))
    if i % 10 == 0
        @info "Finished $i / $N_samples ($((i / N_samples) * 100)%)"
    end
end

out_dict = @strdict rel_errs rmses max_errs ic_rel_errs ic_rmses ic_max_errs std_norms N_newton_steps prior_times initial_condition_times std_dev_times sampling_times optimization_times mat_nnzs chol_nnzs sqmahals nlls
out_dict = merge(out_dict, parameters, @strdict to)

@tagsave(datadir("sims", "burgers", "gmrf-collocation", savename(parameters, "jld2")), out_dict)
