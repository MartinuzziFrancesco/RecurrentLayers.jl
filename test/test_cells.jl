using RecurrentLayers, Flux, Test

#cells returning a single hidden state
single_cells = [AntisymmetricRNNCell, CFNCell, GatedAntisymmetricRNNCell,
    IndRNNCell, LiGRUCell, LightRUCell, MGUCell, MUT1Cell, MUT2Cell,
    MUT3Cell, STARCell]

#cells returning hidden state as a tuple
double_cells = [JANETCell, NASCell, PeepholeLSTMCell, RANCell]

#cells with a little more complexity to them
different_cells = [FastGRNNCell, FastRNNCell, RHNCell, SCRNCell, MinimalRNNCell]

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

@testset "TRNNCell" begin
    rnncell = TRNNCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 2

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, zeros(Float32, 5))

    rnncell = TRNNCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 1

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, zeros(Float32, 5))
end

@testset "TGRUCell" begin
    rnncell = TGRUCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 3)))

    rnncell = TGRUCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 2

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 3)))
end

@testset "TLSTMCell" begin
    rnncell = TLSTMCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) ==
          rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5), zeros(Float32, 3)))

    rnncell = TLSTMCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 2

    inp = rand(Float32, 3)
    @test rnncell(inp) ==
          rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5), zeros(Float32, 3)))
end

@testset "UnICORNNCell" begin
    rnncell = UnICORNNCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 4

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = UnICORNNCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
end

@testset "MinimalRNNCell" begin
    rnncell = MinimalRNNCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 5

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = MinimalRNNCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 4

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = MinimalRNNCell(3 => 5; bias=false, encoder_bias=false)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
end
