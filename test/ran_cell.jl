using Test
using RecurrentLayers
using Flux

@testset "RANCell" begin
    @testset "Sizes and parameters" begin
        rancell = RANCell(3 => 5)
        @test length(Flux.trainables(rancell)) == 3

        inp = rand(Float32, 3)
        @test rancell(inp) == rancell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

        rancell = RANCell(3 => 5; bias=false)
        @test length(Flux.trainables(rancell)) == 2

        inp = rand(Float32, 3)
        @test rancell(inp) == rancell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
    end
end

#@testset "LightRU" begin
#end