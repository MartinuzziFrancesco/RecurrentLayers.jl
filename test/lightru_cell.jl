using Test
using RecurrentLayers
using Flux

@testset "LightRUCell" begin
    @testset "Sizes and parameters" begin
        lightru = LightRUCell(3 => 5)
        @test length(Flux.trainables(lightru)) == 3

        inp = rand(Float32, 3)
        @test lightru(inp) == lightru(inp, zeros(Float32, 5))

        lightru = LightRUCell(3 => 5; bias=false)
        @test length(Flux.trainables(lightru)) == 2

        inp = rand(Float32, 3)
        @test lightru(inp) == lightru(inp, zeros(Float32, 5))
    end
end

#@testset "LightRU" begin
#end