using LinearMaps, SparseArrays, LinearAlgebra

export TridiagonalCholeskyFactor, tridiagonal_cholesky

struct TridiagonalCholeskyFactor{T}
    N::Int
    chos::Vector{<:Cholesky{T}}
    Cs::Vector{AbstractArray{T}}
end

@views function make_chunks(X::AbstractVector, n::Integer)
    c = length(X) ÷ n
    return [X[1+c*k:(k == n - 1 ? end : c * k + c)] for k = 0:n-1]
end

function backward_solve(L::Cholesky, b)
    return L.u \ b
end

function backward_solve(L::SparseArrays.CHOLMOD.Factor, b)
    return L.UP \ b
end

function backward_solve(L::TridiagonalCholeskyFactor, b)
    N_blocks = length(L.chos)
    b_chunks = make_chunks(b, N_blocks)
    x = copy(b_chunks)
    x[end] = backward_solve(L.chos[end], b_chunks[end])
    for i = N_blocks-1:-1:1
        x[i] = backward_solve(L.chos[i], b_chunks[i] - L.Cs[i]' * x[i+1])
    end
    return x
end

function forward_solve(L::Cholesky, b)
    return L.L \ b
end

function forward_solve(L::SparseArrays.CHOLMOD.Factor, b)
    return L.PtL \ b
end

function forward_solve(L::TridiagonalCholeskyFactor, b)
    N_blocks = length(L.chos)
    b_chunks = make_chunks(b, N_blocks)
    x = copy(b_chunks)
    x[1] = forward_solve(L.chos[1], b_chunks[1])
    for i = 2:N_blocks
        x[i] = forward_solve(L.chos[i], b_chunks[i] - L.Cs[i-1] * x[i-1])
    end
    return x
end

function ldiv!(y, L::TridiagonalCholeskyFactor, b)
    x = forward_solve(L, b)
    y .= backward_solve(L, x)
    return y
end

function ldiv(L::TridiagonalCholeskyFactor, b)
    y = similar(b)
    return ldiv!(y, L, b)
end

function tridiagonal_cholesky(A::SparseMatrixCSC, N_blocks)
    block_size = size(A, 1) ÷ N_blocks
    chos = [cholesky(Array(A[1:block_size, 1:block_size]))]
    Cs = AbstractArray{eltype(A)}[]

    for i = 2:N_blocks
        cur_row_start = (i-1) * block_size + 1
        cur_row_stop = i * block_size
        B = Array(A[cur_row_start:cur_row_stop, (i-2)*block_size+1:(i-1)*block_size])
        C = forward_solve(chos[end], B')'
        push!(Cs, C)
        D = Array(A[cur_row_start:cur_row_stop, cur_row_start:cur_row_stop])
        cho = cholesky(D - C * C')
        push!(chos, cho)
        println("Block $i done")
    end
    return TridiagonalCholeskyFactor(size(A, 1), chos, Cs)
end