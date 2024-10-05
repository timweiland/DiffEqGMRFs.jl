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

@info params

const to = TimerOutput()

###### Read data ######
path = datadir("input_data", "Darcy_241", "$datasetname.mat")
ds = DarcyDataset(path)
example_soln, example_coeff = get_problem(ds, 2)
x_coords, y_coords = ds.x_coords, ds.y_coords

###### Form discretization ######
uniform_unit_square_discretization(N_xy; element_order=2) # Trigger precompilation
@timeit to "Mesh generation" disc = uniform_unit_square_discretization(N_xy; element_order=2)

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
        @info "Finished $i / $N_samples ($((i / N_samples) * 100)%)"
    end
end

out_dict = @strdict rel_errs rmses max_errs solve_times pde_disc_times
out_dict = merge(out_dict, params, @strdict to)

@tagsave(datadir("sims", "darcy", "fem", savename(params, "jld2")), out_dict)
