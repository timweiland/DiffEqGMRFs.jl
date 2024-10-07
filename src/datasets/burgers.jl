using MAT
import Base: show, length

export BurgersDataset, get_initial_condition, get_solution

struct BurgersDataset
    burgers_vars::Dict{String}
    x_coords::AbstractVector{Float64}
    ts::StepRangeLen
    ν::Float64

    function BurgersDataset(path)
        burgers_vars = matread(path)
        x_coords = range(0.0, 1.0, Base.size(burgers_vars["input"], 2))
        ts = range(0.0, 1.0, Base.size(burgers_vars["output"], 2))
        ν = burgers_vars["visc"]
        return new(burgers_vars, x_coords, ts, ν)
    end
end

function show(io::IO, ds::BurgersDataset)
    println(
        io,
        "BurgersDataset with $(Base.size(ds.burgers_vars["output"], 1)) samples of size $(Base.size(ds.burgers_vars["output"], 2))x$(Base.size(ds.burgers_vars["output"], 3))",
    )
end

function length(ds::BurgersDataset)
    return Base.size(ds.burgers_vars["output"], 1)
end

function get_initial_condition(ds::BurgersDataset, idx)
    return ds.burgers_vars["input"][idx, :]
end

function get_solution(ds::BurgersDataset, idx)
    return ds.burgers_vars["output"][idx, :, :]
end
