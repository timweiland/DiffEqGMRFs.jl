using Ferrite

export get_periodic_constraint

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