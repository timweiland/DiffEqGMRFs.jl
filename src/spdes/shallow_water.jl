using GaussianMarkovRandomFields, Ferrite, SparseArrays, LinearAlgebra, LinearMaps, SpecialFunctions

export LinearShallowWaterSPDE, discretize

struct LinearShallowWaterSPDE <: SPDE
    H::Function
    τ::Float64
    k::Float64
    f::Float64
    g::Float64

    function LinearShallowWaterSPDE(H = x -> 1.0; τ = 1.0, k = 0.0, f = 0.0, g = 9.81)
        new(H, τ, k, f, g)
    end
end

function assemble_system!(K, M, S, dh, ch, ip, cvh, cvu, cvv, H, k, f, g)
    K_assembler = start_assemble(K)
    M_assembler = start_assemble(M)
    S_assembler = start_assemble(S)
    ke = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    me = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    se = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh)) # stiffness
    range_h = dof_range(dh, :h)
    ndofs_h = length(range_h)
    range_u = dof_range(dh, :u)
    ndofs_u = length(range_u)
    range_v = dof_range(dh, :v)
    ndofs_v = length(range_v)

    ϕₕ = Vector{Float64}(undef, ndofs_h)
    ∇ϕₕ = Vector{Vec{2,Float64}}(undef, ndofs_h)
    ϕᵤ = Vector{Float64}(undef, ndofs_u)
    ∇ϕᵤ = Vector{Vec{2,Float64}}(undef, ndofs_u)
    ϕᵥ = Vector{Float64}(undef, ndofs_v)
    ∇ϕᵥ = Vector{Vec{2,Float64}}(undef, ndofs_v)

    global_dofs = zeros(Int, ndofs_per_cell(dh))

    for cell in CellIterator(dh)
        reinit!(cvh, cell)
        reinit!(cvu, cell)
        reinit!(cvv, cell)
        ke .= 0
        me .= 0
        se .= 0

        cell_coords = getcoordinates(cell)
        for qp = 1:getnquadpoints(cvh)
            x = spatial_coordinate(cvh, qp, cell_coords)
            H_val = H(x)
            dΩ = getdetJdV(cvh, qp)
            for i = 1:ndofs_h
                ϕₕ[i] = shape_value(cvh, qp, i)
                ∇ϕₕ[i] = shape_gradient(cvh, qp, i)
            end
            for i = 1:ndofs_u
                ϕᵤ[i] = shape_value(cvu, qp, i)
                ∇ϕᵤ[i] = shape_gradient(cvu, qp, i)
            end
            for i = 1:ndofs_v
                ϕᵥ[i] = shape_value(cvv, qp, i)
                ∇ϕᵥ[i] = shape_gradient(cvv, qp, i)
            end

            # h - h
            for (i, I) in pairs(range_h), (j, J) in pairs(range_h)
                me[I, J] += ϕₕ[i] * ϕₕ[j] * dΩ
                se[I, J] += ∇ϕₕ[i] ⋅ ∇ϕₕ[j] * dΩ
            end
            # h - u
            for (i, I) in pairs(range_h), (j, J) in pairs(range_u)
                ke[I, J] += -H_val * ∇ϕₕ[i][1] * ϕᵤ[j] * dΩ
            end
            # h - v
            for (i, I) in pairs(range_h), (j, J) in pairs(range_v)
                ke[I, J] += -H_val * ∇ϕₕ[i][2] * ϕᵥ[j] * dΩ
            end

            # u - h
            for (i, I) in pairs(range_u), (j, J) in pairs(range_h)
                ke[I, J] += -g * ∇ϕᵤ[i][1] * ϕₕ[j] * dΩ
            end
            # u - u
            for (i, I) in pairs(range_u), (j, J) in pairs(range_u)
                summand = ϕᵤ[i] * ϕᵤ[j] * dΩ
                me[I, J] += summand
                ke[I, J] += k * summand
                se[I, J] += ∇ϕᵤ[i] ⋅ ∇ϕᵤ[j] * dΩ
            end
            # u - v
            for (i, I) in pairs(range_u), (j, J) in pairs(range_v)
                ke[I, J] += -f * ϕᵤ[i] * ϕᵥ[j] * dΩ
            end

            # v - h
            for (i, I) in pairs(range_v), (j, J) in pairs(range_h)
                ke[I, J] += -g * ∇ϕᵥ[i][2] * ϕₕ[j] * dΩ
            end
            # v - u
            for (i, I) in pairs(range_v), (j, J) in pairs(range_u)
                ke[I, J] += f * ϕᵥ[i] * ϕᵤ[j] * dΩ
            end
            # v - v
            for (i, I) in pairs(range_v), (j, J) in pairs(range_v)
                summand = ϕᵥ[i] * ϕᵥ[j] * dΩ
                me[I, J] += summand
                ke[I, J] += k * summand
                se[I, J] += ∇ϕᵥ[i] ⋅ ∇ϕᵥ[j] * dΩ
            end
        end
        celldofs!(global_dofs, cell)
        assemble!(K_assembler, global_dofs, ke)

        me = lump_matrix(me, ip)
        assemble!(M_assembler, global_dofs, me)
        assemble!(S_assembler, global_dofs, se)
    end
    apply!(K, zeros(size(K, 1)), ch)
    apply!(M, zeros(size(M, 1)), ch)
    apply!(S, zeros(size(S, 1)), ch)
end

function discretize(
    𝒟::LinearShallowWaterSPDE,
    spatial_disc::FEMDiscretization{2},
    ts;
    κ_matern = 1.0,
    mean_offset = 0.0,
    solver_blueprint=CGSolverBlueprint(),
)
    if Set(spatial_disc.dof_handler.field_names) != Set([:h, :u, :v])
        throw(ArgumentError("Expected fields: h, u, v"))
    end
    qr = spatial_disc.quadrature_rule
    ip = spatial_disc.interpolation

    cvh = CellScalarValues(qr, ip)
    cvu = CellScalarValues(qr, ip)
    cvv = CellScalarValues(qr, ip)

    K = create_sparsity_pattern(spatial_disc.dof_handler, spatial_disc.constraint_handler)
    M = create_sparsity_pattern(
        spatial_disc.dof_handler,
        spatial_disc.constraint_handler;
        coupling = [true false false; false true false; false false true],
    )
    G = create_sparsity_pattern(
        spatial_disc.dof_handler,
        spatial_disc.constraint_handler;
        coupling = [true false false; false true false; false false true],
    )

    assemble_system!(
        K,
        M,
        G,
        spatial_disc.dof_handler,
        spatial_disc.constraint_handler,
        spatial_disc.interpolation,
        cvh,
        cvu,
        cvv,
        𝒟.H,
        𝒟.k,
        𝒟.f,
        𝒟.g,
    )

    M̃ = M
    f = spzeros(size(K, 1))
    for dof in spatial_disc.constraint_handler.prescribed_dofs
        G[dof, dof] = 1.0
        M̃[dof, dof] = 1e-2 # TODO
    end

    M̃⁻¹ = spdiagm(0 => 1 ./ diag(M̃))

    K_matern = (κ_matern^2 * M̃ + G)
    # apply!(K_matern, f, spatial_disc.constraint_handler)

    ν = 2
    σ²_natural = gamma(ν) / (gamma(ν + 1) * (4π) * κ_matern^(2 * ν))
    σ²_goal = 1.0
    ratio = σ²_natural / σ²_goal

    # Q_matern = ratio * K_matern * M̃⁻¹ * K_matern * M̃⁻¹ * K_matern
    Q_matern = ratio * K_matern' * M̃⁻¹ * K_matern
    M̃⁻¹_sqrt = spdiagm(0 => sqrt.(1 ./ diag(M̃)))
    Q_matern_sqrt = sqrt(ratio) * K_matern' * M̃⁻¹_sqrt
    # for idx in spatial_disc.constraint_handler.prescribed_dofs
    #     Q_matern[idx, idx] = 1e10 # very certain :D
    # end
    Q₀ = LinearMapWithSqrt(LinearMap(Symmetric(Q_matern)), LinearMap(Q_matern_sqrt))

    x₀ = GMRF(spzeros(size(Q_matern, 1)), Q₀)

    noise_mat = spdiagm(0 => fill(𝒟.τ, Base.size(M, 2)))

    Nₛ = Base.size(K, 2)
    total_ndofs = Nₛ * length(ts)
    mean_offset = fill(mean_offset, total_ndofs)
    for dof in spatial_disc.constraint_handler.prescribed_dofs
        noise_mat[dof, dof] = 1e-2
        st_dofs = dof:Nₛ:total_ndofs
        mean_offset[st_dofs] .= 0.0
    end
    inv_noise_mat = spdiagm(0 => 1 ./ diag(noise_mat))

    β = dt -> sqrt(dt) * noise_mat
    β⁻¹ = dt -> (1 / sqrt(dt)) * inv_noise_mat
    G_fn =
        dt -> (
            S_tmp = M̃ + dt * K; apply!(S_tmp, f, spatial_disc.constraint_handler); LinearMap(
                S_tmp,
            )
        )

    ssm = ImplicitEulerSSM(
        x₀,
        G_fn,
        dt -> LinearMap(M̃),
        dt -> LinearMap(M̃⁻¹),
        β,
        β⁻¹,
        x₀,
        ts,
    )

    X = joint_ssm(ssm)
    X = ImplicitEulerConstantMeshSTGMRF(
        X.mean .+ mean_offset,
        X.precision,
        spatial_disc,
        ssm,
        solver_blueprint,
    )
    if length(spatial_disc.constraint_handler.prescribed_dofs) > 0
        return ConstrainedGMRF(X, spatial_disc.constraint_handler)
    end
    return X
end
