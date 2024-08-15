using DiffEqGMRFs
using Test
using Aqua

@testset "DiffEqGMRFs.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DiffEqGMRFs)
    end
    # Write your tests here.
end
