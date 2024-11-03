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

    Wi = init(out * 2, in)
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
    gxs = chunk(Wi * inp, 2, dims=1)
    ghs = chunk(Wh, 2, dims=1)
    bs = chunk(b, 3, dims=1)

    forget_gate = sigmoid_fast.(gxs[1] .+ bs[1])
    reset_gate = sigmoid_fast.(gxs[1] .+ ghs[1]*state .+ bs[2])
    candidate_state = tanh_fast.(
        ghs[2] * (forget_gate .* state) .+ tanh_fast(inp) + bs[3]
    )
    new_state = candidate_state .* forget_gate .+ state .* (1 .- forget_gate)
    return new_state
end

Base.show(io::IO, mut::MUT1Cell) =
    print(io, "MUT1Cell(", size(mut.Wi, 2), " => ", size(mut.Wi, 1) ÷ 2, ")")