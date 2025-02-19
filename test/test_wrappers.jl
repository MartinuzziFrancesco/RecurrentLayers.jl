using RecurrentLayers, Flux, Test

layers = [RNN, GRU, GRUv3, LSTM, MGU, LiGRU, RAN, LightRU, NAS, MUT1, MUT2, MUT3,
    SCRN, PeepholeLSTM, FastRNN, FastGRNN, LEM]

@testset "Sizes for StackedRNN with layer: $layer" for layer in layers
    wrap = StackedRNN(layer, 2 => 4)

    inp = rand(Float32, 2, 3, 1)
    output = wrap(inp)
    @test output isa Array{Float32, 3}
    @test size(output) == (4, 3, 1)

    inp = rand(Float32, 2, 3)
    output = wrap(inp)
    @test output isa Array{Float32, 2}
    @test size(output) == (4, 3)
end
