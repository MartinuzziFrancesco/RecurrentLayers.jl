using RecurrentLayers, Flux, Test

#cells returning a single hidden state
single_cells = [MGUCell, LiGRUCell, IndRNNCell,
    LightRUCell, MUT1Cell, MUT2Cell,
    MUT3Cell, AntisymmetricRNNCell, GatedAntisymmetricRNNCell]

#cells returning hidden state as a tuple
double_cells = [RANCell, NASCell, PeepholeLSTMCell, JANETCell]

#cells with a little more complexity to them
different_cells = [SCRNCell, RHNCell, FastRNNCell, FastGRNNCell]

@testset "Single return cell: cell = $cell" for cell in single_cells
    rnncell = cell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, zeros(Float32, 5))

    rnncell = cell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 2

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, zeros(Float32, 5))
end

@testset "Double return cell: $cell = " for cell in double_cells
    rnncell = cell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = cell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 2

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
end

@testset "SCRNCell" begin
    rnncell = SCRNCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 5

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = SCRNCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 4

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
end

@testset "RHNCell" begin
    rnncell = RHNCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 6

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, zeros(Float32, 5))

    ##TODO rhncell bias is bugged atm
    rnncell = RHNCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, zeros(Float32, 5))
end

@testset "LEMCell" begin
    rnncell = LEMCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 4

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = LEMCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
end

@testset "coRNNCell" begin
    rnncell = coRNNCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 4

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = coRNNCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
end
