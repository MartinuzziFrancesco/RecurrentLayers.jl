#https://proceedings.mlr.press/v37/jozefowicz15.pdf
struct MUT1Cell{I, H, V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer MUT1Cell

"""
    MUT1Cell((in, out)::Pair; init = glorot_uniform, bias = true)
"""
function MUT1Cell((in, out)::Pair;
    init = glorot_uniform,
    bias = true)

    Wi = init(out * 3, in)
    Wh = init(out * 2, out)
    b = create_bias(Wi, bias, 3 * out)

    return MUT1Cell(Wi, Wh, b)
end

function (mut::MUT1Cell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(mut.Wh, 2))
    return mut(inp, state)
end

function (mut::MUT1Cell)(inp::AbstractVecOrMat, state)
    _size_check(mut, inp, 1 => size(mut.Wi,2))
    Wi, Wh, b = mut.Wi, mut.Wh, mut.bias
    #split
    gxs = chunk(Wi * inp, 3, dims=1)
    ghs = chunk(Wh, 2, dims=1)
    bs = chunk(b, 3, dims=1)

    forget_gate = sigmoid_fast.(gxs[1] .+ bs[1])
    reset_gate = sigmoid_fast.(gxs[2] .+ ghs[1]*state .+ bs[2])
    candidate_state = tanh_fast.(
        ghs[2] * (reset_gate .* state) .+ tanh_fast(gxs[3]) .+ bs[3]
    ) #in the paper is tanh(x_t) but dimensionally it cannot work
    new_state = candidate_state .* forget_gate .+ state .* (1 .- forget_gate)
    return new_state
end

Base.show(io::IO, mut::MUT1Cell) =
    print(io, "MUT1Cell(", size(mut.Wi, 2), " => ", size(mut.Wi, 1) ÷ 3, ")")

struct MUT2Cell{I, H, V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer MUT2Cell

"""
    MUT2Cell((in, out)::Pair; init = glorot_uniform, bias = true)
"""
function MUT2Cell((in, out)::Pair;
    init = glorot_uniform,
    bias = true)

    Wi = init(out * 3, in)
    Wh = init(out * 3, out)
    b = create_bias(Wi, bias, 3 * out)

    return MUT2Cell(Wi, Wh, b)
end

function (mut::MUT2Cell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(mut.Wh, 2))
    return mut(inp, state)
end

function (mut::MUT2Cell)(inp::AbstractVecOrMat, state)
    _size_check(mut, inp, 1 => size(mut.Wi,2))
    Wi, Wh, b = mut.Wi, mut.Wh, mut.bias
    #split
    gxs = chunk(Wi * inp, 3, dims=1)
    ghs = chunk(Wh, 3, dims=1)
    bs = chunk(b, 3, dims=1)

    forget_gate = sigmoid_fast.(gxs[1] .+ ghs[1] * state .+ bs[1])
    reset_gate = sigmoid_fast.(gxs[2] .+ ghs[2]*state .+ bs[2])
    candidate_state = tanh_fast.(ghs[3] * (reset_gate .* state) .+ gxs[3] .+ bs[3])
    new_state = candidate_state .* forget_gate .+ state .* (1 .- forget_gate)
    return new_state
end

Base.show(io::IO, mut::MUT2Cell) =
    print(io, "MUT2Cell(", size(mut.Wi, 2), " => ", size(mut.Wi, 1) ÷ 3, ")")