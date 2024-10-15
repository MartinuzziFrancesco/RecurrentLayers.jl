using RecurrentLayers
using Test
using Aqua

@testset "RecurrentLayers.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(RecurrentLayers; ambiguities = false, deps_compat = false)
    end
    # Write your tests here.
end
