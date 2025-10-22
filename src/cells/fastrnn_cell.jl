#https://arxiv.org/abs/1901.02358
@doc raw"""
    FastRNNCell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        init_alpha = 3.0, init_beta = - 3.0,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Fast recurrent neural network cell [Kusupati2018](@cite).
See [`FastRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: the activation function, defaults to `tanh_fast`.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_alpha`: Initializer for the alpha parameter.
  Default is 3.0.
- `init_beta`: Initializer for the beta parameter.
  Default is - 3.0.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.


# Equations

```math
\begin{aligned}
    \tilde{\mathbf{h}}(t) &= \sigma\left( \mathbf{W}_{ih} \mathbf{x}(t) +
        \mathbf{W}_{hh} \mathbf{h}(t-1) + \mathbf{b} \right), \\
    \mathbf{h}(t) &= \sigma(\alpha) \, \tilde{\mathbf{h}}(t) + \sigma(\beta) \, \mathbf{h}(t-1)
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
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct FastRNNCell{I, H, V, W, M, A, B, F} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::M
    alpha::A
    beta::B
    activation::F
end

@layer FastRNNCell

function FastRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int}, activation=tanh_fast;
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        init_alpha=3.0, init_beta=-3.0,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    T = eltype(weight_ih)
    alpha = T(init_alpha) .* ones(T, 1)
    beta = T(init_beta) .* ones(T, 1)
    integration_fn = _integration_fn(integration_mode)
    return FastRNNCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn,
        alpha, beta, activation)
end

function (fastrnn::FastRNNCell)(inp::AbstractVecOrMat, state)
    _size_check(fastrnn, inp, 1 => size(fastrnn.weight_ih, 2))
    proj_ih = dense_proj(fastrnn.weight_ih, inp, fastrnn.bias_ih)
    proj_hh = dense_proj(fastrnn.weight_hh, state, fastrnn.bias_hh)
    candidate_state = fastrnn.activation.(fastrnn.integration_fn(proj_ih, proj_hh))
    alpha = sigmoid_fast.(fastrnn.alpha)
    beta = sigmoid_fast.(fastrnn.beta)
    new_state = @. alpha * candidate_state + beta * state
    return new_state, new_state
end

function initialstates(fastrnn::FastRNNCell)
    return zeros_like(fastrnn.weight_hh, size(fastrnn.weight_hh, 1))
end

function Base.show(io::IO, fastrnn::FastRNNCell)
    print(io, "FastRNNCell(", size(fastrnn.weight_ih, 2),
        " => ", size(fastrnn.weight_ih, 1) ÷ 2, ")")
end

@doc raw"""
    FastRNN(input_size => hidden_size, [activation];
        return_state = false, kwargs...)

Fast recurrent neural network [Kusupati2018](@cite).
See [`FastRNNCell`](@ref) for a layer that processes a single sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: the activation function, defaults to `tanh_fast`.

# Keyword arguments

- `return_state`: Option to return the last state together with the output.
  Default is `false`.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_alpha`: Initializer for the alpha parameter.
  Default is 3.0.
- `init_beta`: Initializer for the beta parameter.
  Default is - 3.0.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.


# Equations

```math
\begin{aligned}
    \tilde{\mathbf{h}}(t) &= \sigma\left( \mathbf{W}_{ih} \mathbf{x}(t) +
        \mathbf{W}_{hh} \mathbf{h}(t-1) + \mathbf{b} \right), \\
    \mathbf{h}(t) &= \sigma(\alpha) \, \tilde{\mathbf{h}}(t) + \sigma(\beta) \, \mathbf{h}(t-1)
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
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct FastRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand FastRNN

function FastRNN((input_size, hidden_size)::Pair{<:Int, <:Int}, activation=tanh_fast;
        return_state::Bool=false, kwargs...)
    cell = FastRNNCell(input_size => hidden_size, activation; kwargs...)
    return FastRNN{return_state, typeof(cell)}(cell)
end

function functor(rnn::FastRNN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> FastRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, fastrnn::FastRNN)
    print(io, "FastRNN(", size(fastrnn.cell.weight_ih, 2),
        " => ", size(fastrnn.cell.weight_ih, 1))
    print(io, ", ", fastrnn.cell.activation)
    print(io, ")")
end

@doc raw"""
    FastGRNNCell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true, alt_bias=true,
        independent_recurrence = false, integration_mode = :addition)

Fast gated recurrent neural network cell [Kusupati2018](@cite).
See [`FastGRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: the activation function, defaults to `tanh_fast`.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_zeta`: Initializer for the zeta parameter.
  Default is 1.0.
- `init_nu`: Initializer for the nu parameter.
  Default is - 4.0.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `alt_bias`: include different bias for the two gates. Default is `true`
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.


# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{z} \right), \\
    \tilde{\mathbf{h}}(t) &= \tanh\left( \mathbf{W}^{h}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{h}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{h} \right), \\
    \mathbf{h}(t) &= \left( \left( \sigma(\zeta) (1 - \mathbf{z}(t)) + 'sigma(\nu) \right)
        \odot \tilde{\mathbf{h}}(t) \right) + \mathbf{z}(t) \odot \mathbf{h}(t-1)
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
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct FastGRNNCell{I, H, V, W, L, M, A, B, F} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    bias_alt::L
    integration_fn::M
    zeta::A
    nu::B
    activation::F
end

@layer FastGRNNCell

function FastGRNNCell((input_size, hidden_size)::Pair, activation=tanh_fast;
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        init_zeta=1.0, init_nu=-4.0,
        bias::Bool=true, recurrent_bias::Bool=true, alt_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size)
    bias_alt = create_bias(weight_ih, alt_bias, 2 * size(weight_ih, 1))
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    T = eltype(weight_ih)
    zeta = T(init_zeta) .* ones(T, 1)
    nu = T(init_nu) .* ones(T, 1)
    integration_fn = _integration_fn(integration_mode)
    return FastGRNNCell(weight_ih, weight_hh, bias_ih, bias_hh, bias_alt, integration_fn,
        zeta, nu, activation)
end

function (fastgrnn::FastGRNNCell)(inp::AbstractVecOrMat, state)
    _size_check(fastgrnn, inp, 1 => size(fastgrnn.weight_ih, 2))
    proj_ih = dense_proj(fastgrnn.weight_ih, inp, fastgrnn.bias_ih)
    proj_hh = dense_proj(fastgrnn.weight_hh, state, fastgrnn.bias_hh)
    partial_gate = fastgrnn.integration_fn(proj_ih, proj_hh)
    bias_alt_1, bias_alt_2 = chunk(fastgrnn.bias_alt, 2)
    gate = @. fastgrnn.activation(partial_gate + bias_alt_1)
    candidate_state = @. tanh_fast(partial_gate + bias_alt_2)
    t_ones = eltype(gate)(1.0f0)
    zeta = sigmoid_fast.(fastgrnn.zeta)
    nu = sigmoid_fast.(fastgrnn.nu)
    new_state = @. (zeta * (t_ones - gate) + nu) * candidate_state +
                   gate * state
    return new_state, new_state
end

function initialstates(fastgrnn::FastGRNNCell)
    return zeros_like(fastgrnn.weight_hh, size(fastgrnn.weight_hh, 1))
end

function Base.show(io::IO, fastgrnn::FastGRNNCell)
    print(io, "FastGRNNCell(", size(fastgrnn.weight_ih, 2),
        " => ", size(fastgrnn.weight_ih, 1) ÷ 2, ")")
end

@doc raw"""
    FastGRNN(input_size => hidden_size, [activation];
    return_state = false, kwargs...)

Fast recurrent neural network [Kusupati2018](@cite).
See [`FastGRNNCell`](@ref) for a layer that processes a single sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: the activation function, defaults to `tanh_fast`

# Keyword arguments

- `return_state`: Option to return the last state together with the output.
  Default is `false`.
- `init_kernel`: initializer for the input to hidden weights
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_zeta`: Initializer for the zeta parameter.
  Default is 1.0.
- `init_nu`: Initializer for the nu parameter.
  Default is - 4.0.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `alt_bias`: include different bias for the two gates. Default is `true`
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.


# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{z} \right), \\
    \tilde{\mathbf{h}}(t) &= \tanh\left( \mathbf{W}^{h}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{h}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{h} \right), \\
    \mathbf{h}(t) &= \left( \left( \sigma(\zeta) (1 - \mathbf{z}(t)) + 'sigma(\nu) \right)
        \odot \tilde{\mathbf{h}}(t) \right) + \mathbf{z}(t) \odot \mathbf{h}(t-1)
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
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct FastGRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand FastGRNN

function FastGRNN((input_size, hidden_size)::Pair, activation=tanh_fast;
        return_state::Bool=false, kwargs...)
    cell = FastGRNNCell(input_size => hidden_size, activation; kwargs...)
    return FastGRNN{return_state, typeof(cell)}(cell)
end

function functor(rnn::FastGRNN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> FastGRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, fastgrnn::FastGRNN)
    print(io, "FastGRNN(", size(fastgrnn.cell.weight_ih, 2),
        " => ", size(fastgrnn.cell.weight_ih, 1))
    print(io, ", ", fastgrnn.cell.activation)
    print(io, ")")
end
