using Test
using RecurrentLayers
using Flux

@testset "MGUCell" begin
    @testset "Sizes and parameters" begin
        mgu = MGUCell(3 => 5)
        @test length(Flux.trainables(mgu)) == 3

        inp = rand(Float32, 3)
        @test mgu(inp) == mgu(inp, zeros(Float32, 5))

        mgu = MGUCell(3 => 5; bias=false)
        @test length(Flux.trainables(mgu)) == 2

        inp = rand(Float32, 3)
        @test mgu(inp) == mgu(inp, zeros(Float32, 5))
    end
end

#@testset "MGU" begin
#end