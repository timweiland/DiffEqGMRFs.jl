using MAT,
    DiffEqGMRFs,
    GMRFs,
    Ferrite,
    HDF5,
    GLMakie,
    SparseArrays,
    LinearAlgebra,
    LinearMaps,
    Printf,
    TimerOutputs,
    Random
using Distributions
import Base: show

const to = TimerOutput()

struct DarcyDataset
    darcy_vars::Dict{String}
    x_coords::AbstractVector{Float64}
    y_coords::AbstractVector{Float64}

    function DarcyDataset(path)
        darcy_vars = matread(path)
        x_coords = range(0.0, 1.0, Base.size(darcy_vars["sol"], 2))
        y_coords = range(0.0, 1.0, Base.size(darcy_vars["sol"], 3))
        return new(darcy_vars, x_coords, y_coords)
    end
end

function show(io::IO, ds::DarcyDataset)
    println(
        io,
        "DarcyDataset with $(Base.size(ds.darcy_vars["sol"], 1)) samples of size $(Base.size(ds.darcy_vars["sol"], 2))x$(Base.size(ds.darcy_vars["sol"], 3))",
    )
end

function get_problem(ds::DarcyDataset, idx)
    return ds.darcy_vars["sol"][idx, :, :], ds.darcy_vars["coeff"][idx, :, :]
end

function get_xy_idcs(point, x_coords, y_coords)
    x_idx = argmin(abs.(x_coords .- point[1]))
    y_idx = argmin(abs.(y_coords .- point[2]))
    return x_idx, y_idx
end

function assemble_darcy_diff_matrix(
    disc::FEMDiscretization,
    x_coords::AbstractVector,
    y_coords::AbstractVector,
    coeff_mat::AbstractMatrix;
    inflated_boundary = false,
)
    dh = disc.dof_handler

    cellvalues = CellScalarValues(disc.quadrature_rule, disc.interpolation)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0

    G = create_sparsity_pattern(dh)
    f = zeros(ndofs(dh))
    assembler = start_assemble(G, f)
    Ge = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    fe = zeros(ndofs_per_cell(dh))

    keep_dofs = inflated_boundary ? [] : nothing
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        Ge .= 0.0
        fe .= 0.0
        cell_coords = getcoordinates(cell)
        keep = true
        # Loop over quadrature points
        for q_point = 1:getnquadpoints(cellvalues)
            x = spatial_coordinate(cellvalues, q_point, cell_coords)
            if (x[1] < 0.0 || x[1] > 1.0 || x[2] < 0.0 || x[2] > 1.0) && inflated_boundary
                keep = false
            end
            coeff_val = coeff_mat[get_xy_idcs(x, x_coords, y_coords)...]

            # Get the quadrature weight
            dΩ = getdetJdV(cellvalues, q_point)
            # Loop over test shape functions
            for i = 1:n_basefuncs
                δu = shape_value(cellvalues, q_point, i)
                ∇δu = shape_gradient(cellvalues, q_point, i)
                fe[i] += beta * δu * dΩ
                # Loop over trial shape functions
                for j = 1:n_basefuncs
                    ∇u = coeff_val * shape_gradient(cellvalues, q_point, j)
                    # Add contribution to Ke
                    Ge[i, j] += (∇δu ⋅ ∇u) * dΩ
                end
            end
        end
        if keep && inflated_boundary
            push!(keep_dofs, celldofs(cell)...)
        end
        assemble!(assembler, celldofs(cell), Ge, fe)
    end
    ch = disc.constraint_handler
    apply!(G, f, ch)
    return G, f, keep_dofs
end

###### Read data ######
path = "./data/Darcy_241/piececonst_r241_N1024_smooth1.mat"
ds = DarcyDataset(path)
beta = 1.0
example_soln, example_coeff = get_problem(ds, 2)
x_coords, y_coords = ds.x_coords, ds.y_coords

boundary_width = 0.0
inflated_boundary = boundary_width > 0.0

function form_discretization(N_xy; boundary_width = 0.0, use_dirichlet_bc = true)
    x0 = y0 = 0.0 - boundary_width
    x1 = y1 = 1.0 + boundary_width
    grid = generate_grid(
        QuadraticTriangle,
        (N_xy, N_xy),
        Tensors.Vec(x0, y0),
        Tensors.Vec(x1, y1),
    )
    ip = Lagrange{2,RefTetrahedron,2}()
    qr = QuadratureRule{2,RefTetrahedron}(3)

    bcs = []
    if use_dirichlet_bc
        ∂Ω = union(
            getfaceset(grid, "left"),
            getfaceset(grid, "right"),
            getfaceset(grid, "top"),
            getfaceset(grid, "bottom"),
        )
        bc_u = Ferrite.Dirichlet(:u, ∂Ω, (x, t) -> 0.0)
        push!(bcs, bc_u)
    end
    return FEMDiscretization(grid, ip, qr, [(:u, 1)], bcs)
end

###### Form discretization ######
form_discretization(300) # Trigger precompilation
@timeit to "Mesh generation" disc = form_discretization(300)

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
    ν_matern = 2
    desired_range = 0.05
    κ = √(8ν_matern) / desired_range
    spde = MaternSPDE{2}(κ, ν_matern, σ²)

    return GMRFs.discretize(spde, disc)
end

form_prior(disc) # Trigger precompilation
@timeit to "Prior construction" x = form_prior(disc)

@timeit to "Etc" cbp = CholeskySolverBlueprint(RBMCStrategy(50))

##### DARCY OBSERVATIONS ######
function form_observations(
    disc::FEMDiscretization,
    x_coords,
    y_coords,
    example_coeff;
    inflated_boundary = false,
    A_boundary = nothing,
    ys_boundary = nothing,
)
    D, ys_D, keep_dofs = assemble_darcy_diff_matrix(
        disc,
        x_coords,
        y_coords,
        example_coeff;
        inflated_boundary = inflated_boundary,
    )
    if keep_dofs !== nothing
        D = D[keep_dofs, 1:end]
        ys_D = ys_D[keep_dofs]
    end
    A, ys = D, ys_D

    if inflated_boundary
        A = [A; A_boundary]
        ys = [ys; ys_boundary]
    end
    return A, ys
end

A_boundary = ys_boundary = nothing
if inflated_boundary
    boundary_step = 0.003
    boundary_obs = [
        [Tensors.Vec(x, 0.0) for x = 0:boundary_step:1.0]
        [Tensors.Vec(x, 1.0) for x = 0:boundary_step:1.0]
        [Tensors.Vec(0.0, y) for y = 0:boundary_step:1.0]
        [Tensors.Vec(1.0, y) for y = 0:boundary_step:1.0]
    ]
    A_boundary = evaluation_matrix(disc, boundary_obs)
    ys_boundary = zeros(size(A_boundary, 1))
end

A, ys = form_observations(
    disc,
    x_coords,
    y_coords,
    example_coeff;
    inflated_boundary = inflated_boundary,
    A_boundary = A_boundary,
    ys_boundary = ys_boundary,
)
Q_ϵ = 1e8

condition_on_observations(x, A, Q_ϵ, ys; solver_blueprint = cbp) # Trigger precompilation
@timeit to "Conditioning + Node reordering" x_cond =
    condition_on_observations(x, A, Q_ϵ, ys; solver_blueprint = cbp) # For Cholesky permutation
p = x_cond.inner_gmrf.solver_ref[].precision_chol.p

cbp2 = CholeskySolverBlueprint(RBMCStrategy(50), p)

function solve_problem(idx)
    cur_to = TimerOutput()
    example_soln, example_coeff = get_problem(ds, idx)
    @timeit cur_to "PDE Discretization" A, ys = form_observations(
        disc,
        x_coords,
        y_coords,
        example_coeff;
        inflated_boundary = inflated_boundary,
        A_boundary = A_boundary,
        ys_boundary = ys_boundary,
    )
    @timeit cur_to "Conditioning" x_cond =
        condition_on_observations(x, A, Q_ϵ, ys; solver_blueprint = cbp2)
    pred = to_mat(full_mean(x_cond), E, x_coords, y_coords)
    @timeit cur_to "Sampling" full_rand(Random.default_rng(), x_cond)
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

for i = 1:10
    cur_rel_err, cur_rmse, cur_max_err, std_norm, cur_to = solve_problem(i)
    println("Relative error (%) = $(cur_rel_err * 100)")
    println(
        "RMSE: $(@sprintf("%.2e", cur_rmse)), Max error: $(@sprintf("%.2e", cur_max_err))",
    )
    println("Std norm: $(@sprintf("%.2e", std_norm))")
    push!(rel_errs, cur_rel_err)
    push!(rmses, cur_rmse)
    push!(max_errs, cur_max_err)
    push!(std_norms, std_norm)
    push!(conditioning_times, TimerOutputs.time(cur_to["Conditioning"]))
    push!(std_dev_times, TimerOutputs.time(cur_to["Std dev"]))
    push!(sampling_times, TimerOutputs.time(cur_to["Sampling"]))
    push!(pde_disc_times, TimerOutputs.time(cur_to["PDE Discretization"]))
    print_timer(cur_to)
end

out_dict = Dict(
    "Relative error" => rel_errs,
    "RMSE" => rmses,
    "Max error" => max_errs,
    "Std norm" => std_norms,
    "Conditioning times" => conditioning_times,
    "Std dev times" => std_dev_times,
    "Sampling times" => sampling_times,
    "PDE discretization times" => pde_disc_times,
)
