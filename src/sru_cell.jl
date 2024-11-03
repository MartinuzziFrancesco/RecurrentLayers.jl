#https://arxiv.org/pdf/1709.02755
struct SRUCell{I,H,B,V}
    Wi::I
    Wh::H
    v::B
    bias::V
end

Flux.@layer SRUCell

function SRUCell((in, out)::Pair, σ=tanh;
    kernel_init = glorot_uniform,
    recurrent_kernel_init = glorot_uniform,
    bias = true)
    Wi = kernel_init(2 * out, in)
    Wh = recurrent_kernel_init(2 * out, out)
    v = kernel_init(2 * out)
    b = create_bias(Wi, bias, size(Wh, 1))

    return SRUCell(Wi, Wh, v, b)
end

SRUCell(in, out; kwargs...) = SRUCell(in => out; kwargs...)

function (sru::SRUCell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(sru.Wh, 2))
    c_state = zeros_like(state)
    return sru(inp, (state, c_state))
end

function (sru::SRUCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(sru, inp, 1 => size(sru.Wi,2))
    Wi, Wh, v, b = sru.Wi, sru.Wh, sru.v, sru.bias

    #split
    gxs = chunk(Wi * inp, 3, dims=1)
    ghs = chunk(Wh * state .+ b, 2, dims=1)
    vs = chunk(v, 2, dims=1)

    #compute
    input_gate = @. sigmoid_fast(gxs[2] + ghs[1])
    forget_gate = @. sigmoid_fast(gxs[3] + ghs[2])
    candidate_state = @. input_gate * gxs[1] + forget_gate * c_state
    new_state = tanh_fast(candidate_state)
    return new_state, candidate_state
end

Base.show(io::IO, sru::SRUCell) =
    print(io, "SRUCell(", size(sru.Wi, 2), " => ", size(sru.Wi, 1)÷2, ")")