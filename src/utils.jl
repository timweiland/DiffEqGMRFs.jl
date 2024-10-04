using Ferrite, GMRFs

export get_periodic_constraint, uniform_unit_square_discretization

function get_periodic_constraint(grid::Ferrite.Grid{1})
    cellidx_left, dofidx_left = collect(grid.facesets["left"])[1]
    cellidx_right, dofidx_right = collect(grid.facesets["right"])[1]

    temp_dh = DofHandler(grid)
    add!(temp_dh, :u, 1)
    close!(temp_dh)
    cc = CellCache(temp_dh)
    get_dof(cell_idx, dof_idx) = (reinit!(cc, cell_idx); celldofs(cc)[dof_idx])
    dof_left = get_dof(cellidx_left, dofidx_left)
    dof_right = get_dof(cellidx_right, dofidx_right)

    return AffineConstraint(dof_left, [dof_right => 1.0], 0.0)
end

function uniform_unit_square_discretization(N_xy; boundary_width = 0.0, use_dirichlet_bc = true, element_order=2)
    if boundary_width == 0
        grid = create_rectangle(
            0.0,
            0.0,
            1.0,
            1.0,
            1/N_xy;
            element_order=element_order,
        )
    else
        grid = create_inflated_rectangle(
            0.0,
            0.0,
            1.0,
            1.0,
            boundary_width,
            1/N_xy;
            element_order=element_order,
        )
    end
    ip = Lagrange{2,RefTetrahedron,element_order}()
    qr = QuadratureRule{2,RefTetrahedron}(element_order + 1)

    bcs = []
    if use_dirichlet_bc
        ∂Ω = grid.facesets["Interior boundary"]
        bc_u = Ferrite.Dirichlet(:u, ∂Ω, (x, t) -> 0.0)
        push!(bcs, bc_u)
    end
    return FEMDiscretization(grid, ip, qr, [(:u, 1)], bcs)
end
