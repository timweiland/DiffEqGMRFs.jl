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
t_bounds = (0.0, 2.00)
example_problem = burger_set["tensor"][:, :, 3]
CFL = 1.0
ν_burger = CFL * HDF5.attributes(burger_set)["Nu"][] / π

###### Form discretization ######
grid = generate_grid(QuadraticLine, (2*length(x_coords) - 1,), Tensors.Vec(x_start), Tensors.Vec(x_end))
# Every second node corresponds exactly to an x-coordinate in the data
@assert Vector{Float64}(x_coords) ≈ map(n -> n.x[1], grid.nodes[1:2:end])

ip = Lagrange{1,RefCube,2}()
qr = QuadratureRule{1,RefCube}(3)
periodic_constraint = get_periodic_constraint(grid)
disc = FEMDiscretization(grid, ip, qr, [(:u, 1)], [periodic_constraint])

bulk_speed = mean(example_problem[:, 1]) 

ν_matern = 3 // 2
desired_range = 0.001
κ = √(8ν_matern) / desired_range

# c = 1 / (ν_burger / 0.4)
c = 1 / (ν_burger * 0.4)
γ = -c * bulk_speed
spde = AdvectionDiffusionSPDE{1}(0., 1 // 1, 1.0 * ones(1, 1), [γ], c, 1.0 * sqrt(c), ν_matern, κ)
ts = range(t_bounds[1], t_bounds[2], Base.size(example_problem, 2))

###### Prior ######
x_prior = GMRFs.discretize(spde, disc, ts; mean_offset=bulk_speed, prescribed_noise=1e-8)
cbp = CholeskySolverBlueprint(RBMCStrategy(50))

##### Initial condition ######
#A_ic = node_selection_matrix(disc, 3:2:length(grid.nodes))
A_ic = evaluation_matrix(disc, [Tensors.Vec(Float64(x_coords)) for x_coords in x_coords[2:end]])
A_ic = spatial_to_spatiotemporal(A_ic, 1, length(ts))
y_ic = example_problem[:, 1][2:end]
x_ic = condition_on_observations(x_prior, A_ic, 1e8, y_ic; solver_blueprint=cbp)

##### REPRODUCE EXAMPLE ######
# A_eval = evaluation_matrix(disc, [Tensors.Vec(Float64(x_coords)) for x_coords in x_coords[2:end]])
# A_eval = vcat([spatial_to_spatiotemporal(A_eval, i, length(ts)) for i = 1:length(ts)]...)
# y_eval = reshape(example_problem[2:end, 1:end], (1023*length(ts),))
# x_eval = condition_on_observations(x_prior, A_eval, 1e8, y_eval; solver_blueprint=cbp)


##### BURGERS OBSERVATIONS ######
coll_grid = range(x_start, x_end, length = 2*length(x_coords) - 3)
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

ν_burger = HDF5.attributes(burger_set)["Nu"][] / π

f =
    w ->
        (Aₜ₊₁ * w) - (Aₜ * w) + dt * (Aₜ₊₁ * w) .* (∂uₜ₊₁∂x * w) -
        dt * ν_burger * ∂²uₜ₊₁∂x² * w
f!(y, w) = y .= f(w)
y = spzeros(size(Aₜ, 1))
J_static = Aₜ₊₁ - Aₜ - dt * ν_burger * ∂²uₜ₊₁∂x²
J(w) = J_static + dt * (Diagonal(∂uₜ₊₁∂x * w) * Aₜ₊₁ + Diagonal(Aₜ₊₁ * w) * ∂uₜ₊₁∂x)

ν_map =
    w ->
        ((Aₜ₊₁ * w) - (Aₜ * w) + dt * (Aₜ₊₁ * w) .* (∂uₜ₊₁∂x * w)) ./ (dt * ∂²uₜ₊₁∂x² * w)

Q = sparse(to_matrix(x_ic.inner_gmrf.precision))
noise = 1e8

perm = x_ic.inner_gmrf.solver_ref[].precision_chol.p
function gn_step(x, Qx_prior, obs_diff)
    J_mat = J(x)
    A = Symmetric(Q + noise * J_mat' * J_mat)
    rhs = Qx_prior + noise * J_mat' * Array(J_mat * x + obs_diff)
    A_chol = cholesky(A; perm=perm, check=false)
    return A_chol \ rhs
end

x_prior = Array(mean(x_ic))
xₖ = x_prior

obs_diff = y - f(xₖ)
prior_diff = x_prior - xₖ
last_obj_val = Inf
obj_val = dot(prior_diff, Q * prior_diff) + dot(obs_diff, noise * obs_diff)
Qx_prior = Q * x_prior
println("Objective: $obj_val")

rel_diff = (last_val, cur_val) -> abs(last_val - cur_val) / abs(cur_val)

function calculate_obj(x)
    obs_diff = y - f(x)
    prior_diff = x_prior - x
    obj_val = dot(prior_diff, Q * prior_diff) + dot(obs_diff, noise * obs_diff)
    return obj_val
end

N_steps = 0
while (rel_diff(last_obj_val, obj_val) > 1e-4) && N_steps < 20 
    xₖ = gn_step(xₖ, Qx_prior, obs_diff)

    obs_diff = y - f(xₖ)
    prior_diff = x_prior - xₖ
    last_obj_val = obj_val
    obj_val = dot(prior_diff, Q * prior_diff) + dot(obs_diff, noise * obs_diff)
    println("Objective: $obj_val | Observation diff norm: $(norm(obs_diff))")
    N_steps += 1
end

function extract_blocks(I::Vector{Int}, J::Vector{Int}, V::Vector, block_size)
    p = sortperm(I)
    I = I[p]
    J = J[p]
    V = V[p]

    # Initialize new I, J, V for the smaller matrix
    I_diag_block = Int[]
    J_diag_block = Int[]
    V_diag_block = eltype(V)[]  # Keep the element type of V for the new values

    I_off_diag_block = Int[]
    J_off_diag_block = Int[]
    V_off_diag_block = eltype(V)[]  # Keep the element type of V for the new values

    diag_blocks = []
    off_diag_blocks = []

    cur_diag_block_range = 1:block_size
    cur_off_diag_block_range = (1 - block_size):0

    for idx in 1:length(I)
        i = I[idx]
        while i > cur_diag_block_range[end]
            push!(diag_blocks, (I_diag_block, J_diag_block, V_diag_block))
            if cur_off_diag_block_range[1] > 0
                push!(off_diag_blocks, (I_off_diag_block, J_off_diag_block, V_off_diag_block))
            end
            I_diag_block = Int[]
            J_diag_block = Int[]
            V_diag_block = eltype(V)[]  # Keep the element type of V for the new values
        
            I_off_diag_block = Int[]
            J_off_diag_block = Int[]
            V_off_diag_block = eltype(V)[]  # Keep the element type of V for the new values

            cur_diag_block_range = cur_diag_block_range .+ block_size
            cur_off_diag_block_range = cur_off_diag_block_range .+ block_size
            println("New block range: $cur_diag_block_range")
        end
        j = J[idx]

        # Check if the entry (i, j) falls within the block range
        if i in cur_diag_block_range && j in cur_diag_block_range
            push!(I_diag_block, i - first(cur_diag_block_range) + 1)  # Adjust row index to new block
            push!(J_diag_block, j - first(cur_diag_block_range) + 1)  # Adjust column index to new block
            push!(V_diag_block, V[idx])  # Keep the value as is
        end

        if i in cur_diag_block_range && j in cur_off_diag_block_range
            push!(I_off_diag_block, i - first(cur_diag_block_range) + 1)  # Adjust row index to new block
            push!(J_off_diag_block, j - first(cur_off_diag_block_range) + 1)  # Adjust column index to new block
            push!(V_off_diag_block, V[idx])  # Keep the value as is
        end
    end
    push!(diag_blocks, (I_diag_block, J_diag_block, V_diag_block))
    if cur_off_diag_block_range[1] > 0
        push!(off_diag_blocks, (I_off_diag_block, J_off_diag_block, V_off_diag_block))
    end

    diag_blocks = [sparse(I_block, J_block, V_block, block_size, block_size) for (I_block, J_block, V_block) in diag_blocks]
    off_diag_blocks = [sparse(I_block, J_block, V_block, block_size, block_size) for (I_block, J_block, V_block) in off_diag_blocks]
    return diag_blocks, off_diag_blocks
end

J_final = J(xₖ)
# new_precision =
#     precision_map(x_ic) +
#     OuterProductMap(LinearMap(J_final), LinearMaps.UniformScalingMap(noise, size(J_final)[1]))
new_precision = LinearMap(Q + noise * J_final' * J_final)
x_final_inner = ConstantMeshSTGMRF(xₖ, new_precision, disc, x_ic.inner_gmrf.prior.ssm, CholeskySolverBlueprint(RBMCStrategy(100)))
x_final = ConstrainedGMRF(x_final_inner, x_ic.prescribed_dofs, x_ic.free_dofs, x_ic.free_to_prescribed_map, x_ic.free_to_prescribed_offset)
plot_spatiotemporal_gmrf(x_final; compute_std = false)

function mean_pred(X)
    ms = time_means(X)
    pred = zeros(size(example_problem))
    for i in 1:Base.size(pred, 2)
        pred[:, i] = A_coll * ms[i]
    end
    return pred
end

pred = mean_pred(x_final)

comparison_plot(x_coords, example_problem, pred)

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