#https://arxiv.org/pdf/1603.09420
struct MGUCell{I, H, V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer MGUCell

@doc raw"""
    MGUCell((input_size => hidden_size)::Pair;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Minimal gated unit](https://arxiv.org/pdf/1603.09420).
See [`MGU`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
f_t         &= \sigma(W_f x_t + U_f h_{t-1} + b_f), \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (f_t \odot h_{t-1}) + b_h), \\
h_t         &= (1 - f_t) \odot h_{t-1} + f_t \odot \tilde{h}_t
\end{aligned}
```

# Forward

    rnncell(inp, [state])
"""
function MGUCell((input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true)

    Wi = init_kernel(hidden_size * 2, input_size)
    Wh = init_recurrent_kernel(hidden_size * 2, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))

    return MGUCell(Wi, Wh, b)
end

function (mgu::MGUCell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(mgu.Wh, 2))
    return mgu(inp, state)
end

function (mgu::MGUCell)(inp::AbstractVecOrMat, state)
    _size_check(mgu, inp, 1 => size(mgu.Wi,2))
    Wi, Wh, b = mgu.Wi, mgu.Wh, mgu.bias
    #split
    gxs = chunk(Wi * inp .+ b, 2, dims=1)
    ghs = chunk(Wh, 2, dims=1)

    forget_gate = sigmoid_fast.(gxs[1] .+ ghs[1]*state)
    candidate_state = tanh_fast.(gxs[2] .+ ghs[2]*(forget_gate.*state))
    new_state = forget_gate .* state .+ (1 .- forget_gate) .* candidate_state
    return new_state
end

Base.show(io::IO, mgu::MGUCell) =
    print(io, "MGUCell(", size(mgu.Wi, 2), " => ", size(mgu.Wi, 1) ÷ 2, ")")


struct MGU{M}
    cell::M
end
  
Flux.@layer :expand MGU

"""
    MGU((input_size => hidden_size)::Pair; kwargs...)

[Minimal gated unit network](https://arxiv.org/pdf/1603.09420).
See [`MGUCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`
"""
function MGU((input_size, hidden_size)::Pair; kwargs...)
    cell = MGUCell(input_size => hidden_size; kwargs...)
    return MGU(cell)
end

function (mgu::MGU)(inp)
    state = zeros_like(inp, size(mgu.cell.Wh, 2))
    return mgu(inp, state)
end
  
function (mgu::MGU)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = mgu.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end
