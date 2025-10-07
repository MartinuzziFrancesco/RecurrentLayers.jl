#https://arxiv.org/abs/1902.09689
@doc raw"""
    AntisymmetricRNNCell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        independent_recurrence = false, integration_mode = :addition,
        bias = true, recurrent_bias = true,
        epsilon=1.0, gamma = 0.0)


Antisymmetric recurrent cell [Chang2019](@cite).
See [`AntisymmetricRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: activation function. Default is `tanh`.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `epsilon`: step size. Default is 1.0.
- `gamma`: strength of diffusion. Default is 0.0.

# Equations

```math
    \mathbf{h}(t) = \mathbf{h}(t-1) + \epsilon \tanh \left( \mathbf{W}_{ih}
        \mathbf{x}(t) + \left( \mathbf{W}_{hh} - \mathbf{W}_{hh}^\top - \gamma
        \mathbf{I} \right) \mathbf{h}(t-1) + \mathbf{b} \right)
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
struct AntisymmetricRNNCell{F,I,H,V,W,E,G,A} <: AbstractRecurrentCell
    activation::F
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    epsilon::E
    gamma::G
    integration_fn::A
end

@layer AntisymmetricRNNCell

function AntisymmetricRNNCell(
    (input_size, hidden_size)::Pair{<:Int,<:Int}, activation=tanh;
    init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
    bias::Bool=true, recurrent_bias::Bool=true,
    integration_mode::Symbol=:addition, independent_recurrence::Bool=false,
    epsilon::AbstractFloat=1.0f0, gamma::AbstractFloat=0.0f0)
    weight_ih = init_kernel(hidden_size, input_size)
    if independent_recurrence
        @warn "AntisymmetricRNNCell does not support independent_recurrence"
    end
    weight_hh = init_recurrent_kernel(hidden_size, hidden_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    T = eltype(weight_ih)
    integration_fn = _integration_fn(integration_mode)
    return AntisymmetricRNNCell(activation, weight_ih, weight_hh, bias_ih,
        bias_hh, T(epsilon), T(gamma), integration_fn)
end

function (asymrnn::AntisymmetricRNNCell)(
    inp::AbstractArray{T,N}, state::AbstractArray{D,M}) where {T,N,D,M}
    _size_check(asymrnn, inp, 1 => size(asymrnn.weight_ih, 2))
    recurrent_matrix = compute_asym_recurrent(asymrnn.weight_hh, asymrnn.gamma)
    proj_ih = dense_proj(asymrnn.weight_ih, inp, asymrnn.bias_ih)
    proj_hh = dense_proj(recurrent_matrix, state, asymrnn.bias_hh)
    proj_combined = asymrnn.integration_fn(proj_ih, proj_hh)
    new_state = state .+ asymrnn.epsilon .* asymrnn.activation.(proj_combined)
    return new_state, new_state
end

function Base.show(io::IO, asymrnn::AntisymmetricRNNCell)
    print(io, "AntisymmetricRNNCell(", size(asymrnn.weight_ih, 2),
        " => ", size(asymrnn.weight_ih, 1))
    print(io, ", ", asymrnn.activation)
    print(io, ")")
end

@doc raw"""
    AntisymmetricRNN(input_size, hidden_size, [activation];
        return_state = false, kwargs...)

Antisymmetric recurrent neural network [Chang2019](@cite).
See [`AntisymmetricRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: activation function. Default is `tanh`.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `epsilon`: step size. Default is 1.0.
- `gamma`: strength of diffusion. Default is 0.0.

# Equations

```math
    \mathbf{h}(t) = \mathbf{h}(t-1) + \epsilon \tanh \left( \mathbf{W}_{ih}
        \mathbf{x}(t) + \left( \mathbf{W}_{hh} - \mathbf{W}_{hh}^\top - \gamma
        \mathbf{I} \right) \mathbf{h}(t-1) + \mathbf{b} \right)
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
struct AntisymmetricRNN{S,M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand AntisymmetricRNN

function AntisymmetricRNN((input_size, hidden_size)::Pair{<:Int,<:Int}, activation=tanh;
    return_state::Bool=false, kwargs...)
    cell = AntisymmetricRNNCell(input_size => hidden_size, activation; kwargs...)
    return AntisymmetricRNN{return_state,typeof(cell)}(cell)
end

function functor(asymrnn::AntisymmetricRNN{S}) where {S}
    params = (cell=asymrnn.cell,)
    reconstruct = p -> AntisymmetricRNN{S,typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, asymrnn::AntisymmetricRNN)
    print(
        io, "AntisymmetricRNN(", size(asymrnn.cell.weight_ih, 2),
        " => ", size(asymrnn.cell.weight_ih, 1))
    print(io, ", ", asymrnn.cell.activation)
    print(io, ")")
end

@doc raw"""
    GatedAntisymmetricRNNCell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        independent_recurrence = false, integration_mode = :addition,
        bias = true, recurrent_bias = true,
        epsilon=1.0, gamma = 0.0)


Antisymmetric recurrent cell with gating [Chang2019](@cite).
See [`GatedAntisymmetricRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `epsilon`: step size. Default is 1.0.
- `gamma`: strength of diffusion. Default is 0.0.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \left( \mathbf{W}_{hh} - \mathbf{W}_{hh}^\top -
        \gamma \mathbf{I} \right) \mathbf{h}(t-1) + \mathbf{W}^{z}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{z} \right), \\
    \mathbf{h}(t) &= \mathbf{h}(t-1) + \epsilon \, \mathbf{z}(t) \odot
        \tanh\left( \left( \mathbf{W}_{hh} - \mathbf{W}_{hh}^\top - \gamma
        \mathbf{I} \right) \mathbf{h}(t-1) + \mathbf{W}^{h}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{h} \right).
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
struct GatedAntisymmetricRNNCell{I,H,V,W,E,G,A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    epsilon::E
    gamma::G
    integration_fn::A
end

@layer GatedAntisymmetricRNNCell

function GatedAntisymmetricRNNCell(
    (input_size, hidden_size)::Pair{<:Int,<:Int};
    init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
    bias::Bool=true, recurrent_bias::Bool=true,
    integration_mode::Symbol=:addition, independent_recurrence::Bool=false,
    epsilon=1.0f0, gamma=0.0f0)
    weight_ih = init_kernel(hidden_size * 2, input_size)
    if independent_recurrence
        @warn "GatedAntisymmetricRNNCell does not support independent_recurrence"
    end
    weight_hh = init_recurrent_kernel(hidden_size, hidden_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    T = eltype(weight_ih)
    integration_fn = _integration_fn(integration_mode)
    return GatedAntisymmetricRNNCell(
        weight_ih, weight_hh, bias_ih, bias_hh, T(epsilon), T(gamma), integration_fn)
end

function (asymrnn::GatedAntisymmetricRNNCell)(
    inp::AbstractArray{T,N}, state::AbstractArray{D,M}) where {T,N,D,M}
    _size_check(asymrnn, inp, 1 => size(asymrnn.weight_ih, 2))
    recurrent_matrix = compute_asym_recurrent(asymrnn.weight_hh, asymrnn.gamma)
    proj_ih = dense_proj(asymrnn.weight_ih, inp, asymrnn.bias_ih)
    proj_hh = dense_proj(recurrent_matrix, state, asymrnn.bias_hh)
    gxs = chunk(proj_ih, 2; dims=1)
    proj_combined_1 = asymrnn.integration_fn(gxs[1], proj_hh)
    proj_combined_2 = asymrnn.integration_fn(gxs[2], proj_hh)
    input_gate = sigmoid_fast.(proj_combined_1)
    new_state = state .+ asymrnn.epsilon .* input_gate .* tanh_fast.(proj_combined_2)
    return new_state, new_state
end

function Base.show(io::IO, asymrnn::GatedAntisymmetricRNNCell)
    print(io, "GatedAntisymmetricRNNCell(",
        size(asymrnn.weight_ih, 2), " => ", size(asymrnn.weight_ih, 1) ÷ 2)
    print(io, ")")
end

@doc raw"""
    GatedAntisymmetricRNN(input_size, hidden_size;
        return_state = false, kwargs...)

Antisymmetric recurrent neural network with gating [Chang2019](@cite).
See [`GatedAntisymmetricRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `epsilon`: step size. Default is 1.0.
- `gamma`: strength of diffusion. Default is 0.0.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \left( \mathbf{W}_{hh} - \mathbf{W}_{hh}^\top -
        \gamma \mathbf{I} \right) \mathbf{h}(t-1) + \mathbf{W}^{z}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{z} \right), \\
    \mathbf{h}(t) &= \mathbf{h}(t-1) + \epsilon \, \mathbf{z}(t) \odot
        \tanh\left( \left( \mathbf{W}_{hh} - \mathbf{W}_{hh}^\top - \gamma
        \mathbf{I} \right) \mathbf{h}(t-1) + \mathbf{W}^{h}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{h} \right).
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
struct GatedAntisymmetricRNN{S,M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand GatedAntisymmetricRNN

function GatedAntisymmetricRNN((input_size, hidden_size)::Pair{<:Int,<:Int};
    return_state::Bool=false, kwargs...)
    cell = GatedAntisymmetricRNNCell(input_size => hidden_size; kwargs...)
    return GatedAntisymmetricRNN{return_state,typeof(cell)}(cell)
end

function functor(asymrnn::GatedAntisymmetricRNN{S}) where {S}
    params = (cell=asymrnn.cell,)
    reconstruct = p -> GatedAntisymmetricRNN{S,typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, asymrnn::GatedAntisymmetricRNN)
    print(
        io, "GatedAntisymmetricRNN(", size(asymrnn.cell.weight_ih, 2),
        " => ", size(asymrnn.cell.weight_ih, 1) ÷ 2)
    print(io, ")")
end

function compute_asym_recurrent(weight_hh::AbstractArray, gamma::AbstractFloat)
    return weight_hh .- transpose(weight_hh) .-
           gamma .* Matrix{eltype(weight_hh)}(I, size(weight_hh, 1), size(weight_hh, 1))
end
