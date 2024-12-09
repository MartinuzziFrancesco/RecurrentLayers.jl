#https://arxiv.org/abs/1901.02358
struct FastRNNCell{I, H, V, A, B, F}
    Wi::I
    Wh::H
    bias::V
    alpha::A
    beta::B
    activation::F
end

Flux.@layer FastRNNCell

@doc raw"""
    FastRNNCell((input_size => hidden_size)::Pair;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Fast recurrent neural network cell](https://arxiv.org/abs/1901.02358).
See [`FastRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `activation`: the activation function, defaults to `tanh_fast`
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
\tilde{h}_t &= \sigma(W_h x_t + U_h h_{t-1} + b), \\
h_t &= \alpha \tilde{h}_t + \beta h_{t-1}
\end{aligned}
```

# Forward

    fastrnncell(inp, [state])
"""
function FastRNNCell((input_size, hidden_size)::Pair, activation=tanh_fast;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true)

    Wi = init_kernel(hidden_size * 2, input_size)
    Wh = init_recurrent_kernel(hidden_size * 2, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    alpha = 1.f0
    beta = 1.f0

    return FastRNNCell(Wi, Wh, b, alpha, beta, activation)
end

function (fastrnn::FastRNNCell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(mgu.Wh, 2))
    return mgu(inp, state)
end

function (fastrnn::FastRNNCell)(inp::AbstractVecOrMat, state)
    #checks
    _size_check(fastrnn, inp, 1 => size(fastrnn.Wi,2))

    # get variables
    Wi, Wh, b = fastrnn.Wi, fastrnn.Wh, fastrnn.bias
    alpha, beta = fastrnn.alpha, fastrnn.beta

    # perform computations
    candidate_state = fastrnn.activation.(Wi * inp .+ Wh * state .+ b)
    new_state = alpha .* candidate_state .+ beta .* state

    return new_state
end

Base.show(io::IO, fastrnn::FastRNNCell) =
    print(io, "FastRNNCell(", size(fastrnn.Wi, 2), " => ", size(fastrnn.Wi, 1) ÷ 2, ")")


struct FastRNN{M}
    cell::M
end
  
Flux.@layer :expand FastRNN

@doc raw"""
    FastRNN((input_size => hidden_size), [activation]; kwargs...)

[Fast recurrent neural network](https://arxiv.org/abs/1901.02358).
See [`FastRNNCell`](@ref) for a layer that processes a single sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `activation`: the activation function, defaults to `tanh_fast`
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
\tilde{h}_t &= \sigma(W_h x_t + U_h h_{t-1} + b), \\
h_t &= \alpha \tilde{h}_t + \beta h_{t-1}
\end{aligned}
```

# Forward

    fastrnn(inp, [state])
"""
function FastRNN((input_size, hidden_size)::Pair, activation = tanh_fast;
    kwargs...)
    cell = FastRNNCell(input_size => hidden_size, activation; kwargs...)
    return FastRNN(cell)
end

function (fastrnn::FastRNN)(inp)
    state = zeros_like(inp, size(fastrnn.cell.Wh, 2))
    return fastrnn(inp, state)
end
  
function (fastrnn::FastRNN)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = fastrnn.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end
