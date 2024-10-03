export rmse, max_err, rel_err

function rmse(pred, soln)
    return sqrt(mean((pred .- soln).^2))
end

function max_err(pred, soln)
    return maximum(abs.(pred .- soln))
end

function rel_err(pred, soln)
    return norm(pred - soln) / norm(soln)
end