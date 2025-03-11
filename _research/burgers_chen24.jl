using DrWatson
@quickactivate "DiffEqGMRFs"

using Distributions,
    DiffEqGMRFs,
    GMRFs,
    Ferrite,
    SparseArrays,
    LinearAlgebra,
    LinearMaps,
    Printf,
    TimerOutputs,
    Random,
    ArgParse,
    Logging,
    LoggingExtras,
    FastGaussQuadrature

rng = MersenneTwister(675409231)

logger = FormatLogger() do io, args
    if args.level == Logging.Debug
        return
    end
    println(io, "[", args.level, "] ", args.message)
end;

global_logger(logger)

const to = TimerOutput()

###### Argparse ######
function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--N_x"
        help = "Number of FEM basis elements in x direction"
        arg_type = Int
        default = 1000
        "--el_order"
        help = "FEM element oder"
        arg_type = Int
        default = 2
    end
    return parse_args(s)
end
parsed_args = parse_cmd()

el_order = parsed_args["el_order"]
N_x = parsed_args["N_x"]

params = @strdict N_x el_order

@info params

h = 0.001 # spatial grid size
ν = 0.001
T = 1.

# From https://github.com/yifanc96/PDEs-GP-KoleskySolver/blob/main/main_Burgers1d.jl
function sample_points_grid(h_in)
    X_domain = -1+h_in:h_in:1-h_in
    # X_domain = -1:h_in:1
    X_boundary = [-1,1]
    return collect(X_domain), X_boundary
end

function SolveBurgers_ColeHopf(x,t,ν)
    Gauss_pts, weights = gausshermite(100)
    temp = x.-sqrt(4*ν*t)*Gauss_pts
    val1 = weights .* sin.(π*temp) .* exp.(-cos.(π*temp)/(2*π*ν))
    val2 = weights .* exp.(-cos.(π*temp)/(2*π*ν))
    return -sum(val1)/sum(val2)
end

X_domain, X_boundary = sample_points_grid(h)
N_domain = length(X_domain)

function form_prior(disc::FEMDiscretization, ts, ic, N_x, ν_burgers)
    bulk_speed = mean(ic)

    desired_range = sqrt(1 / N_x)

    c = 1 / (ν_burgers)
    γ = -c * bulk_speed
    H = 1.0 * ones(1, 1)
    spde = AdvectionDiffusionSPDE{1}(
        κ = 0.0,
        α = 1 // 1,
        H = H,
        γ = [γ],
        c = c,
        τ = 0.1 * sqrt(c),
        initial_spde = MaternSPDE{1}(range=desired_range, smoothness = 2),
        spatial_spde = MaternSPDE{1}(range=desired_range, smoothness = 1),
    )

    return GMRFs.discretize(spde, disc, ts; mean_offset = bulk_speed, prescribed_noise = 1e-8)
end

function gmrf_solve(; N_x=1000, element_order=2, noise_ic=1e12, noise_fem=1e18)
    grid_shape = (element_order == 2) ? QuadraticLine : Line
    grid = generate_grid(grid_shape, (N_x,), Tensors.Vec(-1.0), Tensors.Vec(1.0))
    ip = Lagrange{RefLine, element_order}()
    qr = QuadratureRule{RefLine}(element_order + 1)
    ∂Ω = grid.facetsets["left"] ∪ grid.facetsets["right"]
    bc = Ferrite.Dirichlet(:u, ∂Ω, (x, t) -> 0.0)
    disc = FEMDiscretization(grid, ip, qr, [(:u, ip)], [bc])

    ts = 0.:0.02:T
    xs = range(-1.0, 1.0, N_x)
    ic = -sin.(π .* X_domain)

    u_prior = form_prior(disc, ts, ic, N_x, ν)

    A_ic = evaluation_matrix(disc, [Tensors.Vec(Float64(x)) for x in X_domain])
    A_ic = spatial_to_spatiotemporal(A_ic, 1, length(ts))

    u_ic = condition_on_observations(u_prior, A_ic, noise_ic, ic)

    begin
        M, G = assemble_burgers_mass_diffusion_matrices(disc; lumping=false)
        Ms = [spatial_to_spatiotemporal(M, t, length(ts)) for t in 1:length(ts)]
        Mₜ = vcat(Ms[1:end-1]...)
        Mₜ₊₁ = vcat(Ms[2:end]...)
        Gs = [spatial_to_spatiotemporal(G, t, length(ts)) for t in 1:length(ts)]
        Gₜ = vcat(Gs[1:end-1]...)
        Gₜ₊₁ = vcat(Gs[2:end]...)
        dt = Float64(step(ts))
        #J_static = Mₜ₊₁ - Mₜ + dt * ν * Gₜ₊₁
        J_static_CN = Mₜ₊₁ - Mₜ + dt * ν * 0.5 * (Gₜ₊₁ + Gₜ)
    end

    N_spatial = ndofs(disc)

    p = u_ic.inner_gmrf.solver_ref[].precision_chol.p
    gncbp = GNCholeskySolverBlueprint(p)

    gno = GaussNewtonOptimizer(
        mean(u_ic),
        precision_map(u_ic),
        x -> f_and_J_CN(x, u_ic, J_static_CN, dt, length(ts), N_spatial, disc),
        noise_fem,
        zeros(size(J_static_CN, 1)),
        mean(u_ic);
        solver_bp=gncbp,
        stopping_criterion=OrCriterion([
            NewtonDecrementCriterion(1e-5),
            StepNumberCriterion(30)
        ])
    )
    optimize(gno)

    J_final = gno.Jₖ
    Q = gno.Q_mat
    new_precision = LinearMap(Q + noise_fem * J_final' * J_final)
    u_final_inner = ImplicitEulerConstantMeshSTGMRF(
        gno.xₖ,
        new_precision,
        disc,
        u_ic.inner_gmrf.prior.ssm,
        CholeskySolverBlueprint(var_strategy=RBMCStrategy(50), perm=p),
    )
    u_final = ConstrainedGMRF(
        u_final_inner,
        u_ic.prescribed_dofs,
        u_ic.free_dofs,
        u_ic.free_to_prescribed_mat,
        u_ic.free_to_prescribed_offset,
    )

    A_eval = evaluation_matrix(disc, [Tensors.Vec(Float64(x)) for x in X_domain])
    A_eval = spatial_to_spatiotemporal(A_eval, length(ts), length(ts))

    sol = A_eval * mean(u_final)
    return sol
end


function nonlinear_primal_tangent(μₖ, disc, N_t, N_spatial)
    Js = SparseMatrixCSC{Float64}[]
    vs = Vector{Float64}[]
    for t in 2:N_t
        cur_μ = μₖ[(t - 1) * N_spatial + 1:t * N_spatial]
        Js_t, vs_t = assemble_burgers_advection_matrix(disc, cur_μ)
        Js_t = spatial_to_spatiotemporal(Js_t, t, N_t)
        push!(Js, Js_t)
        push!(vs, vs_t)
    end
    J = vcat(Js...)
    v = vcat(vs...)
    return v, J
end

function nonlinear_primal_tangent_CN(μₖ, disc, N_t, N_spatial)
    Js = SparseMatrixCSC{Float64}[]
    vs = Vector{Float64}[]
    for t in 1:N_t
        cur_μ = μₖ[(t - 1) * N_spatial + 1:t * N_spatial]
        Js_t, vs_t = assemble_burgers_advection_matrix(disc, cur_μ)
        Js_t = spatial_to_spatiotemporal(Js_t, t, N_t)
        push!(Js, Js_t)
        push!(vs, vs_t)
    end
    Jₜ = vcat(Js[1:end-1]...)
    Jₜ₊₁ = vcat(Js[2:end]...)
    J = 0.5 * (Jₜ + Jₜ₊₁)
    vₜ = vcat(vs[1:end-1]...)
    vₜ₊₁ = vcat(vs[2:end]...)
    v = 0.5 * (vₜ + vₜ₊₁)
    return v, J
end

function f_and_J(w, x_ic, J_static, dt, N_t, N_spatial, disc)
    f_adv, J_adv = nonlinear_primal_tangent(transform_free_to_full(x_ic, w), N_spatial)
    f = J_static * w + dt * f_adv
    J = J_static + dt * J_adv
    return f, J
end

function f_and_J_CN(w, x_ic, J_static_CN, dt, N_t, N_spatial, disc)
    f_adv, J_adv = nonlinear_primal_tangent_CN(transform_free_to_full(x_ic, w), disc, N_t, N_spatial)
    f = J_static_CN * w + dt * f_adv
    J = J_static_CN + dt * J_adv
    return f, J
end


truth =  [SolveBurgers_ColeHopf(X_domain[i],T,ν) for i in 1:N_domain]

@info "First solve..."
gmrf_solve(N_x=N_x, element_order=el_order)
@info "Actual solve..."
@timeit to "Solve time" sol = gmrf_solve(N_x=N_x, element_order=el_order)
@info "Solve done."

err = sol - truth
err_L2 = sqrt(sum(err.^2)/N_domain)
err_MAE = maximum(abs.(err))
err_rel = norm(err) / norm(sol)

out_dict = @strdict err_L2 err_MAE err_rel
@info out_dict
out_dict = merge(out_dict, params, @strdict to)

@tagsave(datadir("sims", "burgers-chen", savename(params, "jld2")), out_dict)
