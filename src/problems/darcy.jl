using Ferrite, GaussianMarkovRandomFields, LinearAlgebra, SparseArrays

export assemble_darcy_diff_matrix

function assemble_darcy_diff_matrix(
    disc::FEMDiscretization,
    x_coords::AbstractVector,
    y_coords::AbstractVector,
    coeff_mat::AbstractMatrix;
    inflated_boundary = false,
    beta=1.0,
    ch=disc.constraint_handler,
)
    dh = disc.dof_handler    

    cellvalues = CellValues(disc.quadrature_rule, disc.interpolation, disc.geom_interpolation)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0

    G = allocate_matrix(dh, ch)
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
    apply!(G, f, ch)
    return G, f, keep_dofs
end
