using Ferrite, GaussianMarkovRandomFields, LinearAlgebra, SparseArrays

export assemble_burgers_advection_matrix, assemble_burgers_mass_diffusion_matrices

function assemble_burgers_advection_matrix(
    disc::FEMDiscretization,
    cur_weights::AbstractVector,
    ch=disc.constraint_handler,
)
    dh = disc.dof_handler

    cellvalues = CellValues(disc.quadrature_rule, disc.interpolation, disc.geom_interpolation)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0

    G = allocate_matrix(dh, ch)
    v = zeros(ndofs(dh))
    assembler = start_assemble(G, v)
    Ge = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    ve = zeros(ndofs_per_cell(dh))

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        Ge .= 0.0
        ve .= 0.0

        w = cur_weights[celldofs(cell)]
        # Loop over quadrature points
        for q_point = 1:getnquadpoints(cellvalues)
            # Get the quadrature weight
            dΩ = getdetJdV(cellvalues, q_point)

            cur_u = Ferrite.function_value(cellvalues, q_point, w)
            # ∇cur_u = Ferrite.function_gradient(cellvalues, q_point, w)[1]
            ∇cur_u = 0.0
            for k = 1:n_basefuncs
                ∇cur_u += shape_gradient(cellvalues, q_point, k)[1] * w[k]
            end
            for i = 1:n_basefuncs
                δu = shape_value(cellvalues, q_point, i)
                # Loop over trial shape functions
                for j = 1:n_basefuncs
                    u = shape_value(cellvalues, q_point, j)
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    # Add contribution to Ke
                    Ge[i, j] += δu * (u * ∇cur_u + cur_u * ∇u[1]) * dΩ
                end
                ve[i] += δu * cur_u * ∇cur_u * dΩ
            end
        end
        assemble!(assembler, celldofs(cell), Ge, ve)
    end
    apply!(G, v, ch)
    v[ch.prescribed_dofs] .= 0.0
    for dof in ch.prescribed_dofs
        G[dof, dof] = 0.0
    end
    return G, v
end

function assemble_burgers_mass_diffusion_matrices(
    disc::FEMDiscretization,
    ch=disc.constraint_handler;
    lumping=false
)
    dh = disc.dof_handler

    cellvalues = CellValues(disc.quadrature_rule, disc.interpolation, disc.geom_interpolation)

    M = allocate_matrix(dh, ch)
    G = allocate_matrix(dh, ch)
    M_assembler = start_assemble(M)
    G_assembler = start_assemble(G)
    Me = spzeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    Ge = spzeros(ndofs_per_cell(dh), ndofs_per_cell(dh))

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        Me .= 0.0
        Ge .= 0.0

        Me = assemble_mass_matrix(Me, cellvalues, disc.interpolation)
        Ge = assemble_diffusion_matrix(Ge, cellvalues)
        assemble!(M_assembler, celldofs(cell), Me)
        assemble!(G_assembler, celldofs(cell), Ge)
    end
    apply!(M, spzeros(size(M, 1)), ch)
    apply!(G, spzeros(size(G, 1)), ch)
    for dof in ch.prescribed_dofs
        M[dof, dof] = 0.0
        G[dof, dof] = 0.0
    end
    if lumping
        M = lump_matrix(M, disc.interpolation)
    end

    return M, G
end
