#https://arxiv.org/pdf/1412.7753
struct SCRNCell{I,H,C,V,A}
    Wi::I
    Wh::H
    Wc::C
    bias::V
    alpha::A
end

Flux.@layer SCRNCell


"""
    SCRNCell(in => out; init = glorot_uniform, bias = true)
"""
function SCRNCell((in, out)::Pair;
    init = glorot_uniform, 
    bias = true,
    alpha = 0.0)

    Wi = init(2 * out, in)
    Wh = init(2 * out, out)
    Wc = init(2 * out, out)
    b = create_bias(Wi, bias, size(Wh, 1))
    return SCRNCell(Wi, Wh, Wc, b, alpha)
end

SCRNCell(in, out; kwargs...) = SCRNCell(in => out; kwargs...)

function (scrn::SCRNCell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(scrn.Wh, 2))
    c_state = zeros_like(state)
    return scrn(inp, (state, c_state))
end

function (scrn::SCRNCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(scrn, inp, 1 => size(scrn.Wi,2))
    Wi, Wh, Wc, b = scrn.Wi, scrn.Wh, scrn.Wc, scrn.bias

    #split
    gxs = chunk(Wi * inp, 2; dims=1)
    ghs = chunk(Wh, 2; dims=1)
    gcs = chunk(Wc * c_state .+ b, 2; dims=1)

    #compute
    context_layer = (1 .- scrn.alpha) .* gxs[1] .+ scrn.alpha .* c_state
    hidden_layer = sigmoid_fast(gxs[2] .+ ghs[1] * state .+ gcs[1])
    new_state = tanh_fast(ghs[2] * hidden_layer .+ gcs[2])
    return new_state, context_layer
end

Base.show(io::IO, scrn::SCRNCell) =
    print(io, "SCRNCell(", size(scrn.Wi, 2), " => ", size(scrn.Wi, 1)÷3, ")")
