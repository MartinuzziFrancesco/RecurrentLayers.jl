#https://arxiv.org/abs/1902.09689
@doc raw"""
    AntisymmetricRNNCell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, epsilon=1.0)


[Antisymmetric recurrent cell](https://arxiv.org/abs/1902.09689).
See [`AntisymmetricRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `activation`: activation function. Default is `tanh`

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`
- `epsilon`: step size. Default is 1.0.
- `gamma`: strenght of diffusion. Default is 0.0

# Equations
```math
h_t = h_{t-1} + \epsilon \tanh \left( (W_h - W_h^T - \gamma I) h_{t-1} + V_h x_t + b_h \right),
```

# Forward

    asymrnncell(inp, state)
    asymrnncell(inp)

## Arguments
- `inp`: The input to the asymrnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the AntisymmetricRNNCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct AntisymmetricRNNCell{F, I, H, V, E, G} <: AbstractRecurrentCell
    activation::F
    Wi::I
    Wh::H
    b::V
    epsilon::E
    gamma::G
end

@layer AntisymmetricRNNCell

function AntisymmetricRNNCell(
        (input_size, hidden_size)::Pair{<:Int, <:Int}, activation=tanh;
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, epsilon=1.0f0, gamma=0.0f0)
    Wi = init_kernel(hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    T = eltype(Wi)
    return AntisymmetricRNNCell(activation, Wi, Wh, b, T(epsilon), T(gamma))
end

function (asymrnn::AntisymmetricRNNCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(asymrnn, inp, 1 => size(asymrnn.Wi, 2))
    Wi, Wh, b = asymrnn.Wi, asymrnn.Wh, asymrnn.b
    epsilon, gamma = asymrnn.epsilon, asymrnn.gamma
    activation = asymrnn.activation
    recurrent_matrix = compute_asym_recurrent(Wh, gamma)
    new_state = state + epsilon .*
                        activation.(Wi * inp .+ recurrent_matrix * state .+ b)
    return new_state, new_state
end

function Base.show(io::IO, asymrnn::AntisymmetricRNNCell)
    print(io, "AntisymmetricRNNCell(", size(asymrnn.Wi, 2), " => ", size(asymrnn.Wi, 1))
    print(io, ", ", asymrnn.activation)
    print(io, ")")
end

@doc raw"""
    AntisymmetricRNN(input_size, hidden_size, [activation];
        return_state = false, kwargs...)

[Antisymmetric recurrent neural network](https://arxiv.org/abs/1902.09689).
See [`AntisymmetricRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `activation`: activation function. Default is `tanh`

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`
- `epsilon`: step size. Default is 1.0.
- `gamma`: strenght of diffusion. Default is 0.0

# Equations
```math
h_t = h_{t-1} + \epsilon \tanh \left( (W_h - W_h^T - \gamma I) h_{t-1} + V_h x_t + b_h \right),
```

# Forward

    asymrnn(inp, state)
    asymrnn(inp)

## Arguments
- `inp`: The input to the asymrnn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the AntisymmetricRNN. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct AntisymmetricRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand AntisymmetricRNN

function AntisymmetricRNN((input_size, hidden_size)::Pair{<:Int, <:Int}, activation=tanh;
        return_state::Bool=false, kwargs...)
    cell = AntisymmetricRNNCell(input_size => hidden_size, activation; kwargs...)
    return AntisymmetricRNN{return_state, typeof(cell)}(cell)
end

function functor(asymrnn::AntisymmetricRNN{S}) where {S}
    params = (cell=asymrnn.cell,)
    reconstruct = p -> AntisymmetricRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, asymrnn::AntisymmetricRNN)
    print(
        io, "AntisymmetricRNN(", size(asymrnn.cell.Wi, 2), " => ", size(asymrnn.cell.Wi, 1))
    print(io, ", ", asymrnn.cell.activation)
    print(io, ")")
end


@doc raw"""
    GatedAntisymmetricRNNCell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, epsilon=1.0)


[Antisymmetric recurrent cell with gating](https://arxiv.org/abs/1902.09689).
See [`GatedAntisymmetricRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`
- `epsilon`: step size. Default is 1.0.
- `gamma`: strenght of diffusion. Default is 0.0

# Equations
```math
\begin{aligned}
    z_t &= \sigma \left( (W_h - W_h^T - \gamma I) h_{t-1} + V_z x_t + b_z \right), \\
    h_t &= h_{t-1} + \epsilon z_t \odot \tanh \left( (W_h - W_h^T - \gamma I) h_{t-1} + V_h x_t + b_h \right).
\end{aligned}
```

# Forward

    asymrnncell(inp, state)
    asymrnncell(inp)

## Arguments
- `inp`: The input to the asymrnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the GatedAntisymmetricRNNCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct GatedAntisymmetricRNNCell{I, H, V, E, G} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    b::V
    epsilon::E
    gamma::G
end

@layer GatedAntisymmetricRNNCell

function GatedAntisymmetricRNNCell(
        (input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, epsilon=1.0f0, gamma=0.0f0)
    Wi = init_kernel(hidden_size * 2, input_size)
    Wh = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    T = eltype(Wi)
    return GatedAntisymmetricRNNCell(Wi, Wh, b, T(epsilon), T(gamma))
end

function (asymrnn::GatedAntisymmetricRNNCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(asymrnn, inp, 1 => size(asymrnn.Wi, 2))
    Wi, Wh, b = asymrnn.Wi, asymrnn.Wh, asymrnn.b
    gxs = chunk(Wi * inp .+ b, 2; dims=1)
    epsilon, gamma = asymrnn.epsilon, asymrnn.gamma
    recurrent_matrix = compute_asym_recurrent(Wh, gamma)
    input_gate = sigmoid_fast(recurrent_matrix * state .+ gxs[1])
    new_state = state + epsilon .* input_gate .*
                        tanh_fast.(gxs[2] .+ recurrent_matrix * state)
    return new_state, new_state
end

function Base.show(io::IO, asymrnn::GatedAntisymmetricRNNCell)
    print(io, "GatedAntisymmetricRNNCell(", size(asymrnn.Wi, 2), " => ", size(asymrnn.Wi, 1) ÷ 2)
    print(io, ")")
end

@doc raw"""
    GatedAntisymmetricRNN(input_size, hidden_size;
        return_state = false, kwargs...)

[Antisymmetric recurrent neural network with gating](https://arxiv.org/abs/1902.09689).
See [`GatedAntisymmetricRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`
- `epsilon`: step size. Default is 1.0.
- `gamma`: strenght of diffusion. Default is 0.0

# Equations
```math
\begin{aligned}
    z_t &= \sigma \left( (W_h - W_h^T - \gamma I) h_{t-1} + V_z x_t + b_z \right), \\
    h_t &= h_{t-1} + \epsilon z_t \odot \tanh \left( (W_h - W_h^T - \gamma I) h_{t-1} + V_h x_t + b_h \right).
\end{aligned}
```

# Forward

    asymrnn(inp, state)
    asymrnn(inp)

## Arguments
- `inp`: The input to the asymrnn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the GatedAntisymmetricRNN. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct GatedAntisymmetricRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand GatedAntisymmetricRNN

function GatedAntisymmetricRNN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = GatedAntisymmetricRNNCell(input_size => hidden_size; kwargs...)
    return GatedAntisymmetricRNN{return_state, typeof(cell)}(cell)
end

function functor(asymrnn::GatedAntisymmetricRNN{S}) where {S}
    params = (cell=asymrnn.cell,)
    reconstruct = p -> GatedAntisymmetricRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, asymrnn::GatedAntisymmetricRNN)
    print(
        io, "GatedAntisymmetricRNN(", size(asymrnn.cell.Wi, 2), " => ", size(asymrnn.cell.Wi, 1) ÷ 2)
    print(io, ")")
end

function compute_asym_recurrent(Wh, gamma)
    return Wh .- transpose(Wh) .- gamma .* Matrix{eltype(Wh)}(I, size(Wh, 1), size(Wh, 1))
end