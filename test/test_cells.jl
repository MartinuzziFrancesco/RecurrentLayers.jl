using RecurrentLayers, Flux, Test

#cells returning a single hidden state
single_cells = [AntisymmetricRNNCell, ATRCell, BRCell, CFNCell, GatedAntisymmetricRNNCell,
    IndRNNCell, LiGRUCell, LightRUCell,
    MiRU1Cell, MiRU2Cell, MGUCell, MUT1Cell, MUT2Cell,
    MUT3Cell, NBRCell, SGRNCell, SGUCell, STARCell, UGRNNCell]

#cells returning hidden state as a tuple
double_cells = [JANETCell, MCLSTMCell, NASCell, OriginalLSTMCell, RANCell]

#cells with a little more complexity to them
different_cells = [FastGRNNCell, FastRNNCell, RHNCell, SCRNCell, MinimalRNNCell]

@testset "Single return cell: cell = $cell" for cell in single_cells
    rnncell = cell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 4

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, zeros(Float32, 5))

    rnncell = cell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, zeros(Float32, 5))
end

@testset "DSGUCell" begin
    rnncell = DSGUCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 5
    @test size(rnncell.weight_ih) == (10, 3)
    @test size(rnncell.weight_hh) == (10, 5)
    @test size(rnncell.weight_out) == (5, 5)

    inp = rand(Float32, 3)
    output, state = rnncell(inp)
    @test size(output) == (5,)
    @test size(state) == (5,)

    rnncell = DSGUCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 4
end

@testset "Double return cell: $cell = " for cell in double_cells
    rnncell = cell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 4

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = cell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
end

@testset "SCRNCell" begin
    rnncell = SCRNCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 7

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = SCRNCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 6

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
    @test length(Flux.trainables(rnncell)) == 6

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = LEMCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 5

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
end

@testset "coRNNCell" begin
    rnncell = coRNNCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 6
    @test rnncell.dt == 0.1f0
    @test rnncell.gamma == 1.0f0
    @test rnncell.epsilon == 1.0f0

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = coRNNCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 5

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    weight_ih = Float32[0.1 -0.2 0.3; 0.4 0.5 -0.6]
    weight_hh = Float32[0.2 -0.1; 0.3 0.4]
    weight_ch = Float32[-0.3 0.7; 0.6 -0.2]
    rnncell = coRNNCell(3 => 2, 0.25f0; gamma=1.5f0, epsilon=0.5f0,
        init_kernel=(args...) -> weight_ih,
        init_recurrent_kernel=(args...) -> weight_hh,
        init_cell_kernel=(args...) -> weight_ch,
        bias=false, recurrent_bias=false, cell_bias=false)
    inp = Float32[0.2, -0.5, 0.7]
    state = Float32[0.6, -0.4]
    c_state = Float32[-0.3, 0.8]
    official_cstate = c_state .+
                      rnncell.dt .*
                      (tanh.(weight_ih * inp .+ weight_hh * state .+ weight_ch * c_state) .-
                       rnncell.gamma .* state .- rnncell.epsilon .* c_state)
    official_state = state .+ rnncell.dt .* official_cstate
    output, new_state = rnncell(inp, (state, c_state))
    @test output ≈ official_state
    @test new_state[1] ≈ official_state
    @test new_state[2] ≈ official_cstate
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
    @test length(Flux.trainables(rnncell)) == 4

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 3)))

    rnncell = TGRUCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 3)))
end

@testset "TauGRUCell" begin
    rnncell = TauGRUCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 4
    @test size(rnncell.weight_ih) == (20, 3)
    @test size(rnncell.weight_hh) == (20, 5)

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = TauGRUCell(3 => 5; delay=2)
    inp = rand(Float32, 3)
    output, state = rnncell(inp)
    @test size(output) == (5,)
    @test size(state[1]) == (5,)
    @test size(state[2]) == (10,)

    inp = rand(Float32, 3, 2)
    output, state = rnncell(inp)
    @test size(output) == (5, 2)
    @test size(state[1]) == (5, 2)
    @test size(state[2]) == (10, 2)

    rnncell = TauGRUCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 3

    @test_throws ArgumentError TauGRUCell(3 => 5; delay=0)
end

@testset "TLSTMCell" begin
    rnncell = TLSTMCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 4

    inp = rand(Float32, 3)
    @test rnncell(inp) ==
          rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5), zeros(Float32, 3)))

    rnncell = TLSTMCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) ==
          rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5), zeros(Float32, 3)))
end

@testset "UnICORNNCell" begin
    rnncell = UnICORNNCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 4
    @test rnncell.dt == 0.1f0
    @test rnncell.alpha == 0.0f0
    @test size(rnncell.weight_hh) == (5,)
    @test size(rnncell.weight_ch) == (5,)
    @test all(x -> 0.0f0 <= x <= 1.0f0, rnncell.weight_hh)
    @test all(x -> -0.1f0 <= x <= 0.1f0, rnncell.weight_ch)
    @test rnncell.bias_hh === false

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = UnICORNNCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    weight_ih = Float32[0.1 -0.2 0.3; 0.4 0.5 -0.6]
    weight_hh = Float32[0.2, 0.7]
    weight_ch = Float32[-0.1, 0.3]
    rnncell = UnICORNNCell(3 => 2, 0.2f0; alpha=0.4f0,
        init_kernel=(args...) -> weight_ih,
        init_recurrent_kernel=(args...) -> weight_hh,
        init_control_kernel=(args...) -> weight_ch,
        bias=false, recurrent_bias=false)
    inp = Float32[0.2, -0.5, 0.7]
    state = Float32[0.6, -0.4]
    c_state = Float32[-0.3, 0.8]
    step = 1.0f0 ./ (1.0f0 .+ exp.(-weight_ch))
    official_cstate = c_state .-
                      rnncell.dt .* step .*
                      (tanh.(state .* weight_hh .+ weight_ih * inp) .+
                       rnncell.alpha .* state)
    official_state = state .+ rnncell.dt .* step .* official_cstate
    output, new_state = rnncell(inp, (state, c_state))
    @test output ≈ official_state
    @test new_state[1] ≈ official_state
    @test new_state[2] ≈ official_cstate
end

@testset "MinimalRNNCell" begin
    rnncell = MinimalRNNCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 6

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = MinimalRNNCell(3 => 5; encoder_bias=false)
    @test length(Flux.trainables(rnncell)) == 5

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
end

@testset "MultiplicativeLSTMCell" begin
    rnncell = MultiplicativeLSTMCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 6

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = MultiplicativeLSTMCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 5

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
end

@testset "WMCLSTMCell" begin
    rnncell = WMCLSTMCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 6

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = WMCLSTMCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 5

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
end

@testset "PeepholeLSTMCell" begin
    rnncell = PeepholeLSTMCell(3 => 5)
    @test length(Flux.trainables(rnncell)) == 6

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))

    rnncell = PeepholeLSTMCell(3 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 5

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, (zeros(Float32, 5), zeros(Float32, 5)))
end

@testset "ResLSTMCell" begin
    rnncell = ResLSTMCell(3 => 5; memory_size=7)
    @test length(Flux.trainables(rnncell)) == 10
    @test size(rnncell.weight_ih) == (26, 3)
    @test size(rnncell.weight_hh) == (26, 5)
    @test size(rnncell.weight_ph) == (14,)
    @test size(rnncell.weight_poh) == (5, 7)
    @test size(rnncell.weight_proj) == (5, 7)
    @test size(rnncell.weight_res) == (5, 3)

    inp = rand(Float32, 3)
    state = (zeros(Float32, 5), zeros(Float32, 7))
    output, new_state = rnncell(inp, state)
    @test size(output) == (5,)
    @test size(new_state[1]) == (5,)
    @test size(new_state[2]) == (7,)
    @test rnncell(inp) == rnncell(inp, state)

    rnncell = ResLSTMCell(3 => 5; bias=false, memory_size=7)
    @test length(Flux.trainables(rnncell)) == 9

    inp = rand(Float32, 3)
    @test rnncell(inp) == rnncell(inp, state)

    @test_throws ArgumentError ResLSTMCell(
        3 => 5; memory_size=7, independent_recurrence=true)
end

@testset "IntersectionRNNCell" begin
    rnncell = IntersectionRNNCell(5 => 5)
    @test length(Flux.trainables(rnncell)) == 4

    inp = rand(Float32, 5)
    @test rnncell(inp) == rnncell(inp, zeros(Float32, 5))

    rnncell = IntersectionRNNCell(5 => 5; bias=false)
    @test length(Flux.trainables(rnncell)) == 3

    inp = rand(Float32, 5)
    @test rnncell(inp) == rnncell(inp, zeros(Float32, 5))
end
