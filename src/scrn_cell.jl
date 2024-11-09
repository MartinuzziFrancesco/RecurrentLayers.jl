#https://arxiv.org/pdf/1412.7753
struct SCRNCell{I,H,C,V,A}
    Wi::I
    Wh::H
    Wc::C
    bias::V
    alpha::A
end

Flux.@layer SCRNCell


@doc raw"""
    SCRNCell((in, out)::Pair;
        kernel_init = glorot_uniform,
        recurrent_kernel_init = glorot_uniform,
        bias = true,
        alpha = 0.0)

[Structurally contraint recurrent unit](https://arxiv.org/pdf/1412.7753).

# Arguments

- `in => out`: input and inner dimension of the layer
- `σ`: activation function. Default is `tanh`
- `kernel_init`: initializer for the input to hidden weights
- `recurrent_kernel_init`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`
- `alpha`: structural contraint. Default is 0.0

# Equations
```math
\begin{aligned}
s_t &= (1 - \alpha) W_s x_t + \alpha s_{t-1}, \\
h_t &= \sigma(W_h s_t + U_h h_{t-1} + b_h), \\
y_t &= f(U_y h_t + W_y s_t)
\end{aligned}
```

# Forward

    rnncell(inp, [state, c_state])
"""
function SCRNCell((in, out)::Pair;
    kernel_init = glorot_uniform,
    recurrent_kernel_init = glorot_uniform,
    bias::Bool = true,
    alpha = 0.0)

    Wi = kernel_init(2 * out, in)
    Wh = recurrent_kernel_init(2 * out, out)
    Wc = recurrent_kernel_init(2 * out, out)
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


struct SCRN{M}
    cell::M
end
  
Flux.@layer :expand SCRN

"""
    SCRN((in, out)::Pair; kwargs...)
"""
function SCRN((in, out)::Pair; kwargs...)
    cell = SCRNCell(in => out; kwargs...)
    return SCRN(cell)
end

function (scrn::SCRN)(inp)
    state = zeros_like(inp, size(scrn.cell.Wh, 2))
    return scrn(inp, state)
end
  
function (scrn::SCRN)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = scrn.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end