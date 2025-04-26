using RecurrentLayers, Flux, Test
import Flux: initialstates

layers = [AntisymmetricRNN, CFN, coRNN, FastGRNN, FastRNN, GatedAntisymmetricRNN,
    JANET, LEM, LiGRU, LightRU, MGU, MinimalRNN, MUT1, MUT2, MUT3, NAS, PeepholeLSTM,
    RAN, SCRN, STAR, TGRU, TLSTM, TRNN, UnICORNN, ATR]
#IndRNN handles internal states diffrently
#RHN should be checked more for consistency for initialstates

@testset "Sizes for layer: $layer" for layer in layers
    rlayer = layer(2 => 4)

    # initial states is zero
    state = initialstates(rlayer)
    if state isa AbstractArray
        @test state ≈ zeros(Float32, 4)
    else
        @test state[1] ≈ zeros(Float32, 4)
        if layer == TGRU
            @test state[2] ≈ zeros(Float32, 2)
        else
            @test state[2] ≈ zeros(Float32, 4)
        end
    end

    inp = rand(Float32, 2, 3, 1)
    output = rlayer(inp, state)
    @test output isa Array{Float32, 3}
    @test size(output) == (4, 3, 1)

    inp = rand(Float32, 2, 3)
    output = rlayer(inp, state)
    @test output isa Array{Float32, 2}
    @test size(output) == (4, 3)
end
