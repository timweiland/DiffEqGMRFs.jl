using DiffEqGMRFs, GMRFs, Ferrite, HDF5, GLMakie, SparseArrays, LinearAlgebra, LinearMaps, Printf

function plot_example(x_coords, soln_mat)
    fig = Figure()
    ax = Axis(fig[1, 1])

    cur_vals = Observable(soln_mat[:, 1])

    lines!(ax, x_coords, cur_vals, color = :blue)

    time_slider = Makie.Slider(fig[2, 1], range=1:size(soln_mat, 2), startvalue=1)
    on(time_slider.value) do val
        cur_vals[] = soln_mat[:, val]
    end

    return fig
end

function comparison_plot(x_coords, soln_mat, pred_mat)
    fig = Figure()
    ax = Axis(fig[1, 1])

    cur_vals = Observable(soln_mat[:, 1])
    cur_pred_vals = Observable(pred_mat[:, 1])

    lines!(ax, x_coords, cur_vals, color = :blue)
    lines!(ax, x_coords, cur_pred_vals, color = :red)

    time_slider = Makie.Slider(fig[2, 1], range=1:size(soln_mat, 2), startvalue=1)
    on(time_slider.value) do val
        cur_vals[] = soln_mat[:, val]
        cur_pred_vals[] = pred_mat[:, val]
    end

    return fig
end

###### Read data ######
burger_set = h5open("./data/burger_nu0.01.hdf5", "r")
x_coords = burger_set["x-coordinate"][:]
t_coords = burger_set["t-coordinate"][:]
x_start, x_end = Float64(x_coords[1]), Float64(x_coords[end])
t_bounds = (0.0, 2.0)
example_problem = burger_set["tensor"][:, :, 2]
CFL = 0.4
ν_burger = CFL * HDF5.attributes(burger_set)["Nu"][] / π

###### Form discretization ######
grid = generate_grid(QuadraticLine, (length(x_coords) - 1,), Tensors.Vec(x_start), Tensors.Vec(x_end))
# Every second node corresponds exactly to an x-coordinate in the data
@assert Vector{Float64}(x_coords) ≈ map(n -> n.x[1], grid.nodes[1:2:end])

ip = Lagrange{1,RefCube,2}()
qr = QuadratureRule{1,RefCube}(3)
periodic_constraint = get_periodic_constraint(grid)
disc = FEMDiscretization(grid, ip, qr, [(:u, 1)], [periodic_constraint])

bulk_speed = mean(example_problem[:, 1]) 

ν_matern = 3 // 2
desired_range = 0.05
κ = √(8ν_matern) / desired_range

# c = 1 / (ν_burger / 0.4)
c = 1 / ν_burger
γ = -c * bulk_speed
spde = AdvectionDiffusionSPDE{1}(0., 1 // 1, 1.0 * ones(1, 1), [γ], c, 0.1 * sqrt(c), ν_matern, κ)
ts = range(t_bounds[1], t_bounds[2], Base.size(example_problem, 2))

###### Prior ######
x_prior = GMRFs.discretize(spde, disc, ts; mean_offset=bulk_speed, prescribed_noise=1e-8)
cbp = CholeskySolverBlueprint(RBMCStrategy(50))

##### Initial condition ######
A_ic = node_selection_matrix(disc, 3:2:length(grid.nodes))
A_ic = spatial_to_spatiotemporal(A_ic, 1, length(ts))
y_ic = example_problem[:, 1][2:end]
x_ic = condition_on_observations(x_prior, A_ic, 1e8, y_ic; solver_blueprint=cbp)

##### BURGERS OBSERVATIONS ######
coll_grid = range(x_start, x_end, length = length(x_coords))
coll_grid = [Tensors.Vec(x) for x in coll_grid]
A_coll = evaluation_matrix(disc, coll_grid)
∂u∂x = derivative_matrices(disc, coll_grid; derivative_idcs = [1])[1]
∂²u∂x² = second_derivative_matrices(disc, coll_grid; derivative_idcs = [(1, 1)])[1]

dt = ts[2] - ts[1]
Aₜ = vcat([spatial_to_spatiotemporal(A_coll, i, length(ts)) for i = 1:(length(ts)-1)]...)
Aₜ₊₁ = vcat([spatial_to_spatiotemporal(A_coll, i, length(ts)) for i = 2:length(ts)]...)
∂uₜ₊₁∂x = vcat([spatial_to_spatiotemporal(∂u∂x, i, length(ts)) for i = 2:length(ts)]...)
∂²uₜ₊₁∂x² = vcat([spatial_to_spatiotemporal(∂²u∂x², i, length(ts)) for i = 2:length(ts)]...)
y = spzeros(size(Aₜ, 1))
Aₜ, y = constrainify_linear_system(Aₜ, y, x_ic)
Aₜ₊₁, y = constrainify_linear_system(Aₜ₊₁, y, x_ic)
∂uₜ₊₁∂x, y = constrainify_linear_system(∂uₜ₊₁∂x, y, x_ic)
∂²uₜ₊₁∂x², y = constrainify_linear_system(∂²uₜ₊₁∂x², y, x_ic)

f =
    w ->
        (Aₜ₊₁ * w) - (Aₜ * w) + dt * (Aₜ₊₁ * w) .* (∂uₜ₊₁∂x * w) -
        dt * ν_burger * ∂²uₜ₊₁∂x² * w
f!(y, w) = y .= f(w)
y = spzeros(size(Aₜ, 1))
J_static = Aₜ₊₁ - Aₜ - dt * ν_burger * ∂²uₜ₊₁∂x²
J(w) = J_static + dt * (Diagonal(∂uₜ₊₁∂x * w) * Aₜ₊₁ + Diagonal(Aₜ₊₁ * w) * ∂uₜ₊₁∂x)

Q = sparse(to_matrix(x_ic.inner_gmrf.precision))
noise = 1e8

function gn_step(x, Qx_prior, obs_diff)
    J_mat = J(x)
    A = Symmetric(Q + noise * J_mat' * J_mat)
    rhs = Qx_prior + noise * J_mat' * (J_mat * x + obs_diff)
    A_chol = cholesky(A)
    return A_chol \ Array(rhs)
end

N_steps = 3
x_prior = Array(mean(x_ic))
xₖ = x_prior

obs_diff = y - f(xₖ)
prior_diff = x_prior - xₖ
obj_val = dot(prior_diff, Q * prior_diff) + dot(obs_diff, noise * obs_diff)
Qx_prior = Q * x_prior
println("Objective: $obj_val")

for i = 1:N_steps
    xₖ = gn_step(xₖ, Qx_prior, obs_diff)

    obs_diff = y - f(xₖ)
    prior_diff = x_prior - xₖ
    obj_val = dot(prior_diff, Q * prior_diff) + dot(obs_diff, noise * obs_diff)
    println("Objective: $obj_val")
end

J_final = J(xₖ)
new_precision =
    precision_map(x_ic) +
    OuterProductMap(LinearMap(J_final), LinearMaps.UniformScalingMap(noise, size(J_final)[1]))
x_final_inner = ConstantMeshSTGMRF(xₖ, new_precision, disc, x_ic.inner_gmrf.prior.ssm, CholeskySolverBlueprint(RBMCStrategy(100)))
x_final = ConstrainedGMRF(x_final_inner, x_ic.prescribed_dofs, x_ic.free_dofs, x_ic.free_to_prescribed_map, x_ic.free_to_prescribed_offset)
plot_spatiotemporal_gmrf(x_final; compute_std = true)

function mean_pred(X)
    ms = time_means(X)
    pred = zeros(size(example_problem))
    for i in 1:Base.size(pred, 2)
        pred[:, i] = A_coll * ms[i]
    end
    return pred
end

pred = mean_pred(x_final)

function rmse(pred, soln)
    return sqrt(mean((pred .- soln).^2))
end

function max_err(pred, soln)
    return maximum(abs.(pred .- soln))
end

pred_rmse = rmse(pred, example_problem)
pred_max_err = max_err(pred, example_problem)
# Print formatted in scientific notation
@printf "RMSE: %.1e, Max error: %.1e\n" pred_rmse pred_max_err
println("RMSE: $(@sprintf("%.2e", pred_rmse)), Max error: $(@sprintf("%.2e", pred_max_err))")