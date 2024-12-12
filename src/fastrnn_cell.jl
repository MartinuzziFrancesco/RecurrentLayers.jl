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

initialstates(fastrnn::FastRNNCell) = zeros_like(fastrnn.Wh, size(fastrnn.Wh, 2))

@doc raw"""
    FastRNNCell((input_size => hidden_size), [activation];
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

    Wi = init_kernel(hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    alpha = randn(Float32)
    beta = randn(Float32)

    return FastRNNCell(Wi, Wh, b, alpha, beta, activation)
end

function (fastrnn::FastRNNCell)(inp::AbstractVecOrMat)
    state = initialstates(fastrnn)
    return fastrnn(inp, state)
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

initialstates(fastrnn::FastRNN) = initialstates(fastrnn.cell)

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
    state = initialstates(fastrnn)
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


struct FastGRNNCell{I, H, V, A, B, F}
    Wi::I
    Wh::H
    bias::V
    zeta::A
    nu::B
    activation::F
end

Flux.@layer FastGRNNCell

initialstates(fastgrnn::FastGRNN) = zeros_like(fastgrnn.Wh, size(fastgrnn.Wh, 2))

@doc raw"""
    FastGRNNCell((input_size => hidden_size), [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Fast gated recurrent neural network cell](https://arxiv.org/abs/1901.02358).
See [`FastGRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `activation`: the activation function, defaults to `tanh_fast`
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z), \\
\tilde{h}_t &= \tanh(W_h x_t + U_h h_{t-1} + b_h), \\
h_t &= \big((\zeta (1 - z_t) + \nu) \odot \tilde{h}_t\big) + z_t \odot h_{t-1}
\end{aligned}
```

# Forward

    fastgrnncell(inp, [state])
"""
function FastGRNNCell((input_size, hidden_size)::Pair, activation=tanh_fast;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true)

    Wi = init_kernel(hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, 2 * size(Wi, 1))
    zeta = randn(Float32)
    nu = randn(Float32)

    return FastGRNNCell(Wi, Wh, b, alpha, beta, activation)
end

function (fastgrnn::FastGRNNCell)(inp::AbstractVecOrMat)
    state = initialstates(fastgrnn)
    return fastgrnn(inp, state)
end

function (fastgrnn::FastGRNNCell)(inp::AbstractVecOrMat, state)
    #checks
    _size_check(fastgrnn, inp, 1 => size(fastgrnn.Wi,2))

    # get variables
    Wi, Wh, b = fastgrnn.Wi, fastgrnn.Wh, fastgrnn.bias
    alpha, beta = fastgrnn.alpha, fastgrnn.beta
    bh, bz = chunk(b, 2)
    partial_gate = Wi * inp .+ Wh * state


    # perform computations
    gate = fastgrnn.activation.(partial_gate .+ bz)
    candidate_state = tanh_fast.(partial_gate .+ bh)
    new_state = (zeta .* (ones(size(gate)) .- gate) .+ nu) .* candidate_state .+ gate .* state

    return new_state
end

Base.show(io::IO, fastgrnn::FastGRNNCell) =
    print(io, "FastGRNNCell(", size(fastgrnn.Wi, 2), " => ", size(fastgrnn.Wi, 1) ÷ 2, ")")


struct FastGRNN{M}
    cell::M
end
  
Flux.@layer :expand FastGRNN

initialstates(fastgrnn::FastGRNN) = initialstates(fastgrnn.cell)

@doc raw"""
    FastGRNN((input_size => hidden_size), [activation]; kwargs...)

[Fast recurrent neural network](https://arxiv.org/abs/1901.02358).
See [`FastGRNNCell`](@ref) for a layer that processes a single sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `activation`: the activation function, defaults to `tanh_fast`
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z), \\
\tilde{h}_t &= \tanh(W_h x_t + U_h h_{t-1} + b_h), \\
h_t &= \big((\zeta (1 - z_t) + \nu) \odot \tilde{h}_t\big) + z_t \odot h_{t-1}
\end{aligned}
```

# Forward

    fastgrnn(inp, [state])
"""
function FastGRNN((input_size, hidden_size)::Pair, activation = tanh_fast;
    kwargs...)
    cell = FastGRNNCell(input_size => hidden_size, activation; kwargs...)
    return FastGRNN(cell)
end

function (fastgrnn::FastGRNN)(inp)
    state = initialstates(fastgrnn)
    return fastgrnn(inp, state)
end
  
function (fastgrnn::FastGRNN)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = fastgrnn.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end