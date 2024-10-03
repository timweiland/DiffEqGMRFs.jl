using MAT
import Base: show

export DarcyDataset, get_problem, get_xy_idcs

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
