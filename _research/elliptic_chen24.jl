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
    LoggingExtras
using Tullio

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
        "--N_el_xy"
        help = "Number of FEM basis elements in x/y direction"
        arg_type = Int
        default = 100
        "--el_order"
        help = "FEM element oder"
        arg_type = Int
        default = 2
    end
    return parse_args(s)
end
parsed_args = parse_cmd()

el_order = parsed_args["el_order"]
N_el_xy = parsed_args["N_el_xy"]

params = @strdict N_el_xy el_order

@info params

const to = TimerOutput()

α = 1.0
m = 3
Ω = [[0,1] [0,1]]
# ground truth solution
freq = 600
s = 6
function fun_u(x)
    ans = 0
    @inbounds for k = 1:freq
        ans += sin(pi*k*x[1])*sin(pi*k*x[2])/k^s
        # H^t norm squared is sum 1/k^{2s-2t}, so in H^{s-1/2}
    end
    return ans
end


ks = 1:freq
πks = π .* ks
k_factors = 2 .* ks.^2 .* (pi^2)
k_pows = ks .^ s

function fun_u_new(x)
    @tullio a = sin(πks[i] * x[1]) * sin(πks[i] * x[2]) / k_pows[i]
end

# right hand side
function fun_rhs(x)
    ans = 0
    @inbounds for k = 1:freq
        ans += (2*k^2*pi^2)*sin(pi*k*x[1])*sin(pi*k*x[2])/k^s
    end
    return ans + α*fun_u(x)^m
end

function fun_rhs_new(x)
    @tullio a = k_factors[i] * sin(πks[i] * x[1]) * sin(πks[i] * x[2]) / k_pows[i]
    return a + α * fun_u_new(x)^m
end

# boundary value
function fun_bdy(x)
    return fun_u(x)
end

function sample_points_grid(h_in, h_bd)
    x1l = Ω[1,1]
    x1r = Ω[2,1]
    x2l = Ω[1,2]
    x2r = Ω[2,2]
    x = x1l + h_in:h_in:x1r-h_in
    y = x2l + h_in:h_in:x2r-h_in
    X_domain = reduce(hcat,[[x[i], y[j]] for i in 1:length(x) for j in 1:length(x)])

    l = length(x1l:h_bd:x1r-h_bd)
    X_boundary = vcat([x1l:h_bd:x1r-h_bd x2l*ones(l)], [x1r*ones(l) x2l:h_bd:x2r-h_bd], [x1r:-h_bd:x1l+h_bd x2r*ones(l)], [x1l*ones(l) x2r:-h_bd:x1l+h_bd])
    return X_domain, X_boundary'
end

h_in = 0.01; h_bd = 0.001
X_domain, X_boundary = sample_points_grid(h_in, h_bd)
N_domain = size(X_domain, 2)
truth = [fun_u(X_domain[:,i]) for i in 1:N_domain]
truth_mat = reshape(truth, (99, 99))

function gmrf_fem_solve(; N_el_xy=100, element_order=2, matern_range=0.1, matern_smoothness=1, boundary_noise=1e12)
    grid_shape = (element_order == 2) ? QuadraticTriangle : Triangle
    grid = generate_grid(grid_shape, (N_el_xy, N_el_xy), Tensors.Vec(0., 0.), Tensors.Vec(1., 1.))
    ip = Lagrange{RefTriangle, element_order}()
    qr = QuadratureRule{RefTriangle}(element_order+1)

    disc = FEMDiscretization(grid, ip, qr)
    spde = MaternSPDE{2}(range = matern_range, smoothness=matern_smoothness)

    x = GMRFs.discretize(spde, disc)

    N_boundary = size(X_boundary, 2)
    A_boundary = evaluation_matrix(disc, [Tensors.Vec(X_boundary[:, i]...) for i in 1:N_boundary])
    x_bc = condition_on_observations(x, A_boundary, 1e12, [fun_bdy(X_boundary[:, i]) for i in 1:N_boundary])
    X_eval = [Tensors.Vec(X_domain[:, i]...) for i in 1:N_domain]
    A_eval = evaluation_matrix(disc, X_eval)

    ∂Ω = grid.facetsets["left"] ∪ grid.facetsets["bottom"] ∪ grid.facetsets["right"] ∪ grid.facetsets["top"]
    bc = Ferrite.Dirichlet(:u, ∂Ω, x -> fun_bdy(x))
    ch_tmp = ConstraintHandler(disc.dof_handler)
    add!(ch_tmp, bc)
    close!(ch_tmp)
    prescribed = ch_tmp.prescribed_dofs

    J_static, f_static = assemble_J_diff_and_f(disc, fun_rhs_new, prescribed)

    p = x_bc.solver_ref[].precision_chol.p
    gncbp = GNCholeskySolverBlueprint(p)
    noise_fem = 3e13

    gno = GaussNewtonOptimizer(
        mean(x_bc),
        precision_map(x_bc),
        x -> f_and_J(x, disc, prescribed, J_static, f_static),
        noise_fem,
        zeros(size(J_static, 1)),
        mean(x_bc);
        solver_bp=gncbp,
        stopping_criterion=OrCriterion([
            NewtonDecrementCriterion(1e-5),
            StepNumberCriterion(10)
        ])
    )
    optimize(gno)

    J_final = gno.Jₖ
    Q = gno.Q_mat
    new_precision = LinearMap(Q + noise_fem * J_final' * J_final)
    x_final = GMRF(gno.xₖ, new_precision, CholeskySolverBlueprint(perm=p))

    sol = A_eval * mean(x_final)

    return sol
end

#∂Ω = grid.facetsets["left"] ∪ grid.facetsets["bottom"] ∪ grid.facetsets["right"] ∪ grid.facetsets["top"]
#bc = Ferrite.Dirichlet(:u, ∂Ω, x -> fun_bdy(x))

#prior_pred = A_eval * full_mean(x_bc)
#prior_pred = reshape(prior_pred, (99, 99))

function assemble_J_diff_and_f(
    disc::FEMDiscretization,
    rhs_fn::Function,
    prescribed_dofs,
)
    dh = disc.dof_handler

    cellvalues = CellValues(disc.quadrature_rule, disc.interpolation, disc.geom_interpolation)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0

    J_diff = allocate_matrix(dh, disc.constraint_handler)
    f = zeros(ndofs(dh))
    assembler = start_assemble(J_diff, f)
    Je = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    fe = zeros(ndofs_per_cell(dh))

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        Je .= 0.0
        fe .= 0.0

        cell_coords = getcoordinates(cell)
        # Loop over quadrature points
        for q_point = 1:getnquadpoints(cellvalues)
            # Get the quadrature weight
            dΩ = getdetJdV(cellvalues, q_point)
            x = spatial_coordinate(cellvalues, q_point, cell_coords)
            rhs_val = rhs_fn(x)

            for i = 1:n_basefuncs
                if celldofs(cell)[i] ∈ prescribed_dofs
                    continue
                end
                δu = shape_value(cellvalues, q_point, i)
                ∇δu = shape_gradient(cellvalues, q_point, i)
                # Loop over trial shape functions
                for j = 1:n_basefuncs
                    u = shape_value(cellvalues, q_point, j)
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    # Add contribution to Ke
                    Je[i, j] += (∇u ⋅ ∇δu) * dΩ
                end
                fe[i] += δu * rhs_val * dΩ
            end
        end
        assemble!(assembler, celldofs(cell), Je, fe)
    end
    return J_diff, f
end


function assemble_J_cube(
    disc::FEMDiscretization,
    cur_weights,
    prescribed_dofs,
)
    dh = disc.dof_handler

    cellvalues = CellValues(disc.quadrature_rule, disc.interpolation, disc.geom_interpolation)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0

    J_cube = allocate_matrix(dh, disc.constraint_handler)
    v = zeros(ndofs(dh))
    assembler = start_assemble(J_cube, v)
    Je = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    ve = zeros(ndofs_per_cell(dh))

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        Je .= 0.0
        ve .= 0.0

        w = cur_weights[celldofs(cell)]
        # Loop over quadrature points
        for q_point = 1:getnquadpoints(cellvalues)
            # Get the quadrature weight
            dΩ = getdetJdV(cellvalues, q_point)

            cur_u = Ferrite.function_value(cellvalues, q_point, w)
            cur_u_sq = cur_u^2
            for i = 1:n_basefuncs
                if celldofs(cell)[i] ∈ prescribed_dofs
                    continue
                end
                δu = shape_value(cellvalues, q_point, i)
                # Loop over trial shape functions
                for j = 1:n_basefuncs
                    u = shape_value(cellvalues, q_point, j)
                    # Add contribution to Ke
                    Je[i, j] += 3 * δu * cur_u_sq * u * dΩ
                end
                ve[i] += δu * cur_u^3 * dΩ
            end
        end
        assemble!(assembler, celldofs(cell), Je, ve)
    end
    return J_cube, v
end

function f_and_J(w, disc, prescribed, J_static, f_static)
    J_cube, f_cube = assemble_J_cube(disc, w, prescribed)
    f = J_static * w + f_cube - f_static
    J = J_static + J_cube
    return f, J
end


function plot_w(w)
    pred = A_eval * w
    pred = reshape(pred, (99, 99))
    heatmap(pred)
end

@info "First solve..."
gmrf_fem_solve(N_el_xy=N_el_xy, element_order=el_order)
@info "Actual solve..."
@timeit to "Solve time" sol = gmrf_fem_solve(N_el_xy=N_el_xy, element_order=el_order)
@info "Solve done."

err = truth - sol
err_L2 = sqrt(sum(err.^2)/N_domain)
err_MAE = maximum(abs.(err))
err_rel = norm(err) / norm(sol)


out_dict = @strdict err_L2 err_MAE err_rel
out_dict = merge(out_dict, params, @strdict to)

@tagsave(datadir("sims", "elliptic-chen", savename(params, "jld2")), out_dict)
