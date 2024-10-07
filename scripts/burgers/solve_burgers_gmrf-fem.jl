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
        "--N_x"
        help = "Number of FEM elements"
        arg_type = Int
        default = 800
        "--dry_run"
        help = "Test run which does not go through the entire dataset"
        arg_type = Bool
        default = true
    end
    return parse_args(s)
end
parsed_args = parse_cmd()

rng = MersenneTwister(345903459)
###### Read data ######
datasetname = parsed_args["datasetname"]
N_x = parsed_args["N_x"]
dry_run = parsed_args["dry_run"]
# beta = 1.0
parameters = @strdict datasetname N_x dry_run

@info parameters

const to = TimerOutput()

###### Read data ######
path = datadir("input_data", "Burgers", "$datasetname.mat")
ds = BurgersDataset(path)
example_ic, example_solution = get_initial_condition(ds, 2), get_solution(ds, 2)
x_coords, ts = ds.x_coords, ds.ts

###### Form discretization ######
periodic_unit_interval_discretization(N_x; element_order=2) # Trigger precompilation
@timeit to "Mesh generation" disc = periodic_unit_interval_discretization(N_x; element_order=2)

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
function form_prior(disc::FEMDiscretization, ts, ic, N_x, ν_burgers)
    bulk_speed = mean(ic)

    ν_matern = 3 // 2
    desired_range = sqrt(1 / N_x)
    κ = √(8ν_matern) / desired_range

    c = 1 / (ν_burgers)
    γ = -c * bulk_speed
    spde = AdvectionDiffusionSPDE{1}(
        0.0,
        1 // 1,
        1.0 * ones(1, 1),
        [γ],
        c,
        0.1 * sqrt(c),
        ν_matern,
        κ,
    )

    return GMRFs.discretize(spde, disc, ts; mean_offset = bulk_speed, prescribed_noise = 1e-8)
end

x = form_prior(disc, ts, example_ic, N_x, ds.ν) # Trigger precompilation
@timeit to "Prior construction" x = form_prior(disc, ts, example_ic, N_x, ds.ν)
cbp = CholeskySolverBlueprint(RBMCStrategy(50, rng))

# Runtime for A_ic is negligible (try it)
A_ic = evaluation_matrix(disc, [Tensors.Vec(Float64(x)) for x in x_coords])
A_ic = spatial_to_spatiotemporal(A_ic, 1, length(ts))

### FEM
@timeit to "PDE Discretization (Linear part)" begin
    M, G = assemble_burgers_mass_diffusion_matrices(disc; lumping=false)
    Ms = [spatial_to_spatiotemporal(M, t, length(ts)) for t in 1:length(ts)]
    Mₜ = vcat(Ms[1:end-1]...)
    Mₜ₊₁ = vcat(Ms[2:end]...)
    Gₜ₊₁ = vcat([spatial_to_spatiotemporal(G, t, length(ts)) for t in 2:length(ts)]...)
    dt = Float64(step(ts))
    J_static = Mₜ₊₁ - Mₜ + dt * ds.ν * Gₜ₊₁
end

N_spatial = ndofs(disc)
function nonlinear_primal_tangent(μₖ)
    Js = SparseMatrixCSC{Float64}[]
    vs = Vector{Float64}[]
    for t in 2:length(ts)
        cur_μ = μₖ[(t - 1) * N_spatial + 1:t * N_spatial]
        Js_t, vs_t = assemble_burgers_advection_matrix(disc, cur_μ)
        Js_t = spatial_to_spatiotemporal(Js_t, t, length(ts))
        push!(Js, Js_t)
        push!(vs, vs_t)
    end
    J = vcat(Js...)
    v = vcat(vs...)
    return v, J
end

function f_and_J(w, x_ic)
    f_adv, J_adv = nonlinear_primal_tangent(transform_free_to_full(x_ic, w))
    f = J_static * w + dt * f_adv
    J = J_static + dt * J_adv
    return f, J
end

noise_ic = 1e8
noise_fem = 1e12

function solve_problem(idx)
    cur_to = TimerOutput()
    example_ic, example_soln = get_initial_condition(ds, idx), get_solution(ds, idx)
    example_soln = example_soln[2:end, 1:end]
    ys_ic = example_ic

    @timeit cur_to "Initial condition" x_ic = condition_on_observations(x, A_ic, noise_ic, ys_ic; solver_blueprint = cbp)

    ic_pred = to_mat(full_mean(x_ic), E, ts, x_coords)
    ic_pred = ic_pred[2:end, 1:end]
    ic_rel_err = rel_err(ic_pred, example_soln)
    ic_rmse = rmse(ic_pred, example_soln)
    ic_max_err = max_err(ic_pred, example_soln)

    p = x_ic.inner_gmrf.solver_ref[].precision_chol.p
    gncbp = GNCholeskySolverBlueprint(p)

    gno = GaussNewtonOptimizer(
        mean(x_ic),
        precision_map(x_ic),
        x -> f_and_J(x, x_ic),
        noise_fem,
        zeros(size(J_static, 1)),
        mean(x_ic);
        solver_bp=gncbp,
    )
    @timeit cur_to "Optimization" begin
        optimize(gno)

        J_final = gno.Jₖ
        Q = gno.Q_mat
        new_precision = LinearMap(Q + noise_fem * J_final' * J_final)
        x_final_inner = ImplicitEulerConstantMeshSTGMRF(
            gno.xₖ,
            new_precision,
            disc,
            x_ic.inner_gmrf.prior.ssm,
            CholeskySolverBlueprint(RBMCStrategy(50), p),
        )
        x_final = ConstrainedGMRF(
            x_final_inner,
            x_ic.prescribed_dofs,
            x_ic.free_dofs,
            x_ic.free_to_prescribed_mat,
            x_ic.free_to_prescribed_offset,
        )
    end
    mat_nnz = nnz(to_matrix(precision_map(x_final)))
    chol_nnz = nnz(x_final.inner_gmrf.solver_ref[].precision_chol)
    
    pred = to_mat(full_mean(x_final), E, ts, x_coords)
    pred = pred[2:end, 1:end]
    @timeit cur_to "Sampling" full_rand(rng, x_final)
    @timeit cur_to "Std dev" cur_std = std(x_final)
    cur_rel_err = rel_err(pred, example_soln)
    cur_rmse = rmse(pred, example_soln)
    cur_max_err = max_err(pred, example_soln)
    std_norm = norm(cur_std)
    N_newton_steps = length(gno.r_obs_norm_history) - 1
    return cur_rel_err, cur_rmse, cur_max_err, ic_rel_err, ic_rmse, ic_max_err, std_norm, N_newton_steps, mat_nnz, chol_nnz, cur_to
end

rel_errs = Float64[]
rmses = Float64[]
max_errs = Float64[]
ic_rel_errs = Float64[]
ic_rmses = Float64[]
ic_max_errs = Float64[]
std_norms = Float64[]
N_newton_steps = Int64[]
initial_condition_times = Int64[]
std_dev_times = Int64[]
optimization_times = Int64[]
sampling_times = Int64[]
mat_nnzs = Int64[]
chol_nnzs = Int64[]

N_samples = dry_run ? 3 : length(ds)
@info "Beginning to solve $N_samples problems"
for i = 1:N_samples
    cur_rel_err, cur_rmse, cur_max_err, ic_rel_err, ic_rmse, ic_max_err, std_norm, N_newton_step, mat_nnz, chol_nnz, cur_to = solve_problem(i)
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
    push!(initial_condition_times, TimerOutputs.time(cur_to["Initial condition"]))
    push!(std_dev_times, TimerOutputs.time(cur_to["Std dev"]))
    push!(optimization_times, TimerOutputs.time(cur_to["Optimization"]))
    push!(sampling_times, TimerOutputs.time(cur_to["Sampling"]))
    if i % 10 == 0
        @info "Finished $i / $N_samples ($((i / N_samples) * 100)%)"
    end
end

out_dict = @strdict rel_errs rmses max_errs ic_rel_errs ic_rmses ic_max_errs std_norms N_newton_steps initial_condition_times std_dev_times sampling_times optimization_times mat_nnzs chol_nnzs
out_dict = merge(out_dict, parameters, @strdict to)

@tagsave(datadir("sims", "burgers", "gmrf-fem", savename(parameters, "jld2")), out_dict)
