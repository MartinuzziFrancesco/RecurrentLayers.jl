using Test
using RecurrentLayers
using Flux

@testset "LiGRUCell" begin
    @testset "Sizes and parameters" begin
        ligru = LiGRUCell(3 => 5)
        @test length(Flux.trainables(ligru)) == 3

        inp = rand(Float32, 3)
        @test ligru(inp) == ligru(inp, zeros(Float32, 5))

        ligru = LiGRUCell(3 => 5; bias=false)
        @test length(Flux.trainables(ligru)) == 2

        inp = rand(Float32, 3)
        @test ligru(inp) == ligru(inp, zeros(Float32, 5))
    end
end

#@testset "LiGRU" begin
#end