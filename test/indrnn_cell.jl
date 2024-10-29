using Test
using RecurrentLayers
using Flux

@testset "IndRNNCell" begin
    @testset "Sizes and parameters" begin
        indrnn = IndRNNCell(3 => 5)
        @test length(Flux.trainables(indrnn)) == 3

        inp = rand(Float32, 3)
        @test indrnn(inp) == indrnn(inp, zeros(Float32, 5))

        indrnn = IndRNNCell(3 => 5; bias=false)
        @test length(Flux.trainables(indrnn)) == 2

        inp = rand(Float32, 3)
        @test indrnn(inp) == indrnn(inp, zeros(Float32, 5))
    end
end

#@testset "IndRNN" begin
#end