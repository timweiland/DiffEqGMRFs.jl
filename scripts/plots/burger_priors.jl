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
    MAT,
    CairoMakie,
    DifferentialEquations,
    Pardiso
using Distributions
import Base: show

###### Argparse ######
function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--N_basis"
        help = "Number of FEM basis elements"
        arg_type = Int
        default = 750
        "--prior_type"
        help = "Prior type to use. adv_diff or product_matern"
        arg_type = String
        default = "adv_diff"
        "--spatial_range"
        help = "Spatial range"
        arg_type = Float64
        default = 0.5
        "--matern_temporal_lengthscale"
        help = "Temporal lengthscale for Matern prior"
        arg_type = Float64
        default = 4.0
    end
    return parse_args(s)
end
parsed_args = parse_cmd()

rng = MersenneTwister(985368934)
###### Read data ######
N_basis = parsed_args["N_basis"]
prior_type = parsed_args["prior_type"]
spatial_range = parsed_args["spatial_range"]
matern_temporal_lengthscale = parsed_args["matern_temporal_lengthscale"]

###### Form discretization ######
grid = generate_grid(QuadraticLine, (N_basis,), Tensors.Vec(-6.0), Tensors.Vec(6.0))
ip = Lagrange{RefLine, 2}()
qr = QuadratureRule{RefLine}(3)
bcs = [(get_periodic_constraint(grid), 1e-2)]

disc = FEMDiscretization(grid, ip, qr, [(:u, nothing)], bcs)

###### Prior ######
function form_adv_diff_prior(disc::FEMDiscretization, ts, ic, range, ν_burgers)
    bulk_speed = mean(ic)

    ν_matern = 3 // 2
    desired_range = range
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
    #return spde

    return GaussianMarkovRandomFields.discretize(spde, disc, ts; mean_offset = bulk_speed, prescribed_noise = 1e-8)
end

function form_product_matern_prior(disc::FEMDiscretization, ts, spatial_range, matern_temporal_lengthscale)
    temporal_matern = MaternSPDE{1}(range=matern_temporal_lengthscale, smoothness=0, σ²=0.1)
    spatial_matern = MaternSPDE{1}(range=spatial_range, smoothness=3, σ²=0.1)

    return product_matern(temporal_matern, length(ts), spatial_matern, disc; solver_blueprint = CholeskySolverBlueprint(var_strategy=RBMCStrategy(2000)))
end

function form_prior(disc::FEMDiscretization, ts, ic, spatial_range, ν_burgers, prior_type, matern_temporal_lengthscale)
    if prior_type == "adv_diff"
        return form_adv_diff_prior(disc, ts, ic, spatial_range, ν_burgers)
    elseif prior_type == "product_matern"
        return form_product_matern_prior(disc, ts, spatial_range, matern_temporal_lengthscale)
    else
        error("Unknown prior type: $prior_type")
    end
end

ts = 0:0.02:3
ν = 0.5
x_coords = range(-6.0, 6.0, length=1000)
example_ic = map(x -> 1.0 * exp(-x^2 / 8), x_coords)

x_adv_diff = form_prior(disc, ts, example_ic, spatial_range, ν, "adv_diff", matern_temporal_lengthscale)
x_product_matern = form_prior(disc, ts, example_ic, spatial_range, ν, "product_matern", matern_temporal_lengthscale)

A_ic = evaluation_matrix(disc, [Tensors.Vec(Float64(x)) for x in x_coords])
A_ic = spatial_to_spatiotemporal(A_ic, 1, length(ts))

rng = MersenneTwister(34590349)

plot_points = range(-6.0, 6.0, length=1000)
A_t = evaluation_matrix(disc, [Tensors.Vec(Float64(x)) for x in plot_points])

x_cond_adv_diff = condition_on_observations(x_adv_diff, A_ic, 1e8, example_ic; solver_blueprint=PardisoGMRFSolverBlueprint())
x_cond_product_matern = condition_on_observations(x_product_matern, A_ic, 1e5, example_ic; solver_blueprint=PardisoGMRFSolverBlueprint())

# We approximate u_x using a central difference and u_xx using the standard second order central difference.
function burgers_pde!(du, u, p, t)
    ν, dx = p
    N = length(u)
    # Loop over grid points, using periodic boundary conditions via modular arithmetic
    for i in 1:N
        ip = mod1(i + 1, N)   # index for i+1 with wrap-around
        im = mod1(i - 1, N)   # index for i-1 with wrap-around
        # Central difference for first derivative
        u_x = (u[ip] - u[im]) / (2 * dx)
        # Central difference for second derivative
        u_xx = (u[ip] - 2*u[i] + u[im]) / (dx^2)
        du[i] = -u[i] * u_x + ν * u_xx
    end
end

# Set up the spatial grid
N = 1000                         # Number of spatial grid points
L = 2π                          # Domain length (adjust as needed)
x = LinRange(-6.0, 6.0, N+1)[1:end-1]  # Uniform grid; drop the last point to enforce periodicity
dx = x[2] - x[1]

# Define an initial condition; here we use sin(x) as an example.
u0 = map(x -> 1.0 * exp(-x^2 / 8), x)

# Define the viscosity parameter
p = (ν, dx)

# Define the time span for the simulation
tspan = (ts[1], ts[end])

# Set up and solve the ODE problem
prob = ODEProblem(burgers_pde!, u0, tspan, p)
sol = DifferentialEquations.solve(prob, Rosenbrock23(), tstops=ts)

function compare_at_time(t, x_cond)
    idx = findfirst(x -> x ≈ t, ts)
    sol_idx = findfirst(x -> x ≈ t, sol.t)

    fig = Figure()
    ax = Axis(fig[1, 1], limits=(-6.0, 6.0, -0.08, 1.08))
    plot!(ax, x_cond, idx)
    lines!(ax, x, sol.u[sol_idx], color=:orange, linewidth=5)
    hidedecorations!(ax)
    return fig
end

start_plot = compare_at_time(0.0, x_cond_product_matern)
save("./plots/burger_priors/start.pdf", start_plot)

plot_times = [1.5, 3.0]
for t in plot_times
    plot_matern = compare_at_time(t, x_cond_product_matern)
    save("./plots/burger_priors/matern_$(t).pdf", plot_matern)
    plot_adv_diff = compare_at_time(t, x_cond_adv_diff)
    save("./plots/burger_priors/adv_diff_$(t).pdf", plot_adv_diff)
end

