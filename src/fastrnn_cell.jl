#https://arxiv.org/abs/1901.02358
struct FastRNNCell{I, H, V, A, B, F} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    bias::V
    alpha::A
    beta::B
    activation::F
end

Flux.@layer FastRNNCell

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

    fastrnncell(inp, state)
    fastrnncell(inp)

## Arguments
- `inp`: The input to the fastrnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the FastRNN. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros.

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state `new_state`, 
  a tensor of size `hidden_size` or `hidden_size x batch_size`.
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

function (fastrnn::FastRNNCell)(inp::AbstractVecOrMat, state)
    #checks
    _size_check(fastrnn, inp, 1 => size(fastrnn.Wi,2))

    # get variables
    Wi, Wh, b = fastrnn.Wi, fastrnn.Wh, fastrnn.bias
    alpha, beta = fastrnn.alpha, fastrnn.beta

    # perform computations
    candidate_state = fastrnn.activation.(Wi * inp .+ Wh * state .+ b)
    new_state = alpha .* candidate_state .+ beta .* state

    return new_state, new_state
end

Base.show(io::IO, fastrnn::FastRNNCell) =
    print(io, "FastRNNCell(", size(fastrnn.Wi, 2), " => ", size(fastrnn.Wi, 1) ÷ 2, ")")


struct FastRNN{M} <: AbstractRecurrentLayer
    cell::M
end
  
Flux.@layer :noexpand FastRNN

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

    fastrnn(inp, state)
    fastrnn(inp)

## Arguments
- `inp`: The input to the fastrnn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the FastRNN. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros.

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
"""
function FastRNN((input_size, hidden_size)::Pair, activation = tanh_fast;
    kwargs...)
    cell = FastRNNCell(input_size => hidden_size, activation; kwargs...)
    return FastRNN(cell)
end
  
function (fastrnn::FastRNN)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    return scan(fastrnn.cell, inp, state)
end


struct FastGRNNCell{I, H, V, A, B, F} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    bias::V
    zeta::A
    nu::B
    activation::F
end

Flux.@layer FastGRNNCell

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

    fastgrnncell(inp, state)
    fastgrnncell(inp)

## Arguments

- `inp`: The input to the fastgrnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the FastGRNN. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros.

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state `new_state`, 
  a tensor of size `hidden_size` or `hidden_size x batch_size`.
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

    return FastGRNNCell(Wi, Wh, b, zeta, nu, activation)
end

function (fastgrnn::FastGRNNCell)(inp::AbstractVecOrMat, state)
    #checks
    _size_check(fastgrnn, inp, 1 => size(fastgrnn.Wi,2))

    # get variables
    Wi, Wh, b = fastgrnn.Wi, fastgrnn.Wh, fastgrnn.bias
    zeta, nu = fastgrnn.zeta, fastgrnn.nu
    bh, bz = chunk(b, 2)
    partial_gate = Wi * inp .+ Wh * state


    # perform computations
    gate = fastgrnn.activation.(partial_gate .+ bz)
    candidate_state = tanh_fast.(partial_gate .+ bh)
    new_state = (zeta .* (ones(size(gate)) .- gate) .+ nu) .* candidate_state .+ gate .* state

    return new_state, new_state
end

Base.show(io::IO, fastgrnn::FastGRNNCell) =
    print(io, "FastGRNNCell(", size(fastgrnn.Wi, 2), " => ", size(fastgrnn.Wi, 1) ÷ 2, ")")


struct FastGRNN{M} <: AbstractRecurrentLayer
    cell::M
end
  
Flux.@layer :noexpand FastGRNN

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

    fastgrnn(inp, state)
    fastgrnn(inp)

## Arguments

- `inp`: The input to the fastgrnn. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the FastGRNN. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros.

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
"""
function FastGRNN((input_size, hidden_size)::Pair, activation = tanh_fast;
    kwargs...)
    cell = FastGRNNCell(input_size => hidden_size, activation; kwargs...)
    return FastGRNN(cell)
end