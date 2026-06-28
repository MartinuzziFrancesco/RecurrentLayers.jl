using RecurrentLayers, Flux, Test
import Flux: initialstates

layers = [
    AntisymmetricRNN, ATR, BR, CFN, coRNN, DSGU, FastGRNN, FastRNN, GatedAntisymmetricRNN,
    IndRNN, JANET, LEM, LiGRU, LightRU, MCLSTM, MGU, MinimalRNN, MiRU1, MiRU2,
    MultiplicativeLSTM, MUT1, MUT2, MUT3, NAS, OriginalLSTM, NBR, PeepholeLSTM,
    RAN, ResLSTM, SCRN, SGRN, SGU, STAR, TauGRU, TGRU, TLSTM, TRNN, UGRNN, UnICORNN,
    WMCLSTM]
#IndRNN handles internal states differently
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
        elseif layer == TauGRU
            @test state[2] ≈ zeros(Float32, 4)
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

@testset "Sizes for layer: TauGRU delay" begin
    rlayer = TauGRU(2 => 4; delay=3, return_state=true)
    state = initialstates(rlayer)
    @test state[1] ≈ zeros(Float32, 4)
    @test state[2] ≈ zeros(Float32, 12)

    inp = rand(Float32, 2, 5, 3)
    output, state = rlayer(inp, state)
    @test output isa Array{Float32, 3}
    @test size(output) == (4, 5, 3)
    @test size(state[1]) == (4, 3)
    @test size(state[2]) == (12, 3)
end

@testset "Sizes for layer: ResLSTM with projected memory" begin
    rlayer = ResLSTM(2 => 4; memory_size=6)

    state = initialstates(rlayer)
    @test state[1] ≈ zeros(Float32, 4)
    @test state[2] ≈ zeros(Float32, 6)

    inp = rand(Float32, 2, 3, 1)
    output = rlayer(inp, state)
    @test output isa Array{Float32, 3}
    @test size(output) == (4, 3, 1)

    inp = rand(Float32, 2, 3)
    output = rlayer(inp, state)
    @test output isa Array{Float32, 2}
    @test size(output) == (4, 3)
end

@testset "Sizes for layer: IntersectionRNN" begin
    rlayer = IntersectionRNN(4 => 4)

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

    inp = rand(Float32, 4, 3, 1)
    output = rlayer(inp, state)
    @test output isa Array{Float32, 3}
    @test size(output) == (4, 3, 1)

    inp = rand(Float32, 4, 3)
    output = rlayer(inp, state)
    @test output isa Array{Float32, 2}
    @test size(output) == (4, 3)
end
