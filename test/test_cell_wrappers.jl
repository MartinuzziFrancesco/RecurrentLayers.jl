using RecurrentLayers, Flux, Test

cells = [
    AntisymmetricRNNCell, ATRCell, BRCell, CFNCell, coRNNCell, FastGRNNCell, FastRNNCell,
    GatedAntisymmetricRNNCell, IndRNNCell, JANETCell, LEMCell, LiGRUCell,
    LightRUCell, MGUCell, MinimalRNNCell, MultiplicativeLSTMCell, MUT1Cell, MUT2Cell,
    MUT3Cell, NASCell, OriginalLSTMCell, NBRCell,
    PeepholeLSTMCell, RANCell, SCRNCell, SGRNCell, STARCell,
    TGRUCell,
    TRNNCell, UGRNNCell, UnICORNNCell, WMCLSTMCell]
#RHNCell, RHNCellUnit, FSRNNCell, TLSTMCell

@testset "Sizes for Multiplicative with cell: $cell" for cell in cells
    wrap = Multiplicative(cell, 2 => 4)

    inp = rand(Float32, 2, 3)
    output = wrap(inp)
    @test first(output) isa Array{Float32, 2}
    @test size(first(output)) == (4, 3)

    inp = rand(Float32, 2)
    output = wrap(inp)
    @test first(output) isa Array{Float32, 1}
    @test size(first(output)) == (4,)
end

@testset "Sizes for Recurrence Multiplicative with cell: $cell" for cell in cells
    wrap = Recurrence(Multiplicative(cell, 2 => 4))

    inp = rand(Float32, 2, 3, 1)
    output = wrap(inp)
    @test output isa Array{Float32, 3}
    @test size(output) == (4, 3, 1)

    inp = rand(Float32, 2, 3)
    output = wrap(inp)
    @test output isa Array{Float32, 2}
    @test size(output) == (4, 3)
end
