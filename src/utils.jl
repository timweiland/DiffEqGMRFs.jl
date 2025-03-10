using Ferrite, GaussianMarkovRandomFields

export get_periodic_constraint, uniform_unit_square_discretization, periodic_unit_interval_discretization

function get_periodic_constraint(grid::Ferrite.Grid{1}; element_order=2)
    cellidx_left, dofidx_left = collect(grid.facetsets["left"])[1]
    cellidx_right, dofidx_right = collect(grid.facetsets["right"])[1]

    temp_dh = DofHandler(grid)
    add!(temp_dh, :u, Lagrange{RefLine,element_order}())
    close!(temp_dh)
    cc = CellCache(temp_dh)
    get_dof(cell_idx, dof_idx) = (reinit!(cc, cell_idx); celldofs(cc)[dof_idx])
    dof_left = get_dof(cellidx_left, dofidx_left)
    dof_right = get_dof(cellidx_right, dofidx_right)

    return AffineConstraint(dof_left, [dof_right => 1.0], 0.0)
end

function uniform_unit_square_discretization(N_xy; boundary_width = 0.0, use_dirichlet_bc = true, element_order=2, boundary_noise=1e-2)
    grid = create_inflated_rectangle(
        0.0,
        0.0,
        1.0,
        1.0,
        boundary_width,
        1/N_xy;
        element_order=element_order,
    )
    ip = Lagrange{RefTriangle,element_order}()
    qr = QuadratureRule{RefTriangle}(element_order + 1)

    bcs = []
    if use_dirichlet_bc
        ∂Ω = grid.facetsets["Interior boundary"]
        bc_u = Ferrite.Dirichlet(:u, ∂Ω, (x, t) -> 0.0)
        push!(bcs, (bc_u, boundary_noise))
    end
    return FEMDiscretization(grid, ip, qr, [(:u, nothing)], bcs)
end

function periodic_unit_interval_discretization(N_x; element_order=element_order, boundary_noise=1e-2)
    grid = generate_grid(QuadraticLine, (N_x,), Tensors.Vec(0.0), Tensors.Vec(1.0))
    ip = Lagrange{RefLine, element_order}()
    qr = QuadratureRule{RefLine}(element_order + 1)

    bcs = [(get_periodic_constraint(grid; element_order=element_order), boundary_noise)]
    return FEMDiscretization(grid, ip, qr, [(:u, nothing)], bcs)
end
