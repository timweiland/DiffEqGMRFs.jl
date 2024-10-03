using DrWatson
@quickactivate "DiffEqGMRFs"

using Distributions,
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
    Random,
    ArgParse
import Base: show

###### Argparse ######
function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--datasetname"
        help = "Name of the Darcy flow dataset to use"
        arg_type = String
        default = "piececonst_r241_N1024_smooth1"
        "--N_xy"
        help = "Number of FEM elements in each direction"
        arg_type = Int
        default = 300
        "--dry_run"
        help = "Test run which does not go through the entire dataset"
        arg_type = Bool
        default = false
    end
    return parse_args(s)
end
parsed_args = parse_cmd()

###### Params ######
datasetname = parsed_args["datasetname"]
N_xy = parsed_args["N_xy"]
dry_run = parsed_args["dry_run"]
beta = 1.0
params = @strdict datasetname N_xy dry_run beta

println(params)

const to = TimerOutput()

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
datasetname = "piececonst_r241_N1024_smooth1"
path = datadir("input_data", "Darcy_241", "$datasetname.mat")
ds = DarcyDataset(path)
example_soln, example_coeff = get_problem(ds, 2)
x_coords, y_coords = ds.x_coords, ds.y_coords

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
form_discretization(N_xy) # Trigger precompilation
@timeit to "Mesh generation" disc = form_discretization(N_xy)

@timeit to "Etc" begin
    pred_coords = [Tensors.Vec(Float64(x), Float64(y)) for x in x_coords for y in y_coords]
    E = evaluation_matrix(disc, pred_coords)
end

function to_mat(dof_vals, E, x_coords, y_coords)
    pred = E * dof_vals
    return reshape(pred, (length(x_coords), length(y_coords)))'
end

function solve_problem(idx)
    cur_to = TimerOutput()
    example_soln, example_coeff = get_problem(ds, idx)

    @timeit cur_to "PDE Discretization" K, f, _ = assemble_darcy_diff_matrix(
        disc,
        x_coords,
        y_coords,
        example_coeff;
        inflated_boundary = false,
    )
    @timeit cur_to "Linear solve" (u = K \ f; apply!(u, disc.constraint_handler))
    pred = to_mat(u, E, x_coords, y_coords)
    cur_rel_err = rel_err(pred, example_soln)
    cur_rmse = rmse(pred, example_soln)
    cur_max_err = max_err(pred, example_soln)
    return cur_rel_err, cur_rmse, cur_max_err, cur_to
end

rel_errs = Float64[]
rmses = Float64[]
max_errs = Float64[]
solve_times = Int64[]
pde_disc_times = Int64[]

N_samples = dry_run ? 3 : Base.size(ds.darcy_vars["sol"], 1)
for i = 1:N_samples
    cur_rel_err, cur_rmse, cur_max_err, cur_to = solve_problem(i)
    push!(rel_errs, cur_rel_err)
    push!(rmses, cur_rmse)
    push!(max_errs, cur_max_err)
    push!(pde_disc_times, TimerOutputs.time(cur_to["PDE Discretization"]))
    push!(solve_times, TimerOutputs.time(cur_to["Linear solve"]))
    if i % 10 == 0
        println("Finished $i / $N_samples ($((i / N_samples) * 100)%)")
    end
end

out_dict = @strdict rel_errs rmses max_errs solve_times pde_disc_times
out_dict = merge(out_dict, params)

@tagsave(datadir("sims", "darcy", "fem", savename(params, "jld2")), out_dict)
