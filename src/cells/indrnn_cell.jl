#https://arxiv.org/pdf/1803.04831

@doc raw"""
    IndRNNCell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = true, integration_mode = :addition)


Independently recurrent cell [Li2018](@cite).
See [`IndRNN`](@ref) for a layer that processes entire sequences.

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
  recurrent to recurrent weights are a vector instead of a matrix. Default `true`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.


# Equations

```math
    \mathbf{h}(t) = \sigma\left( \mathbf{W}_{ih} \mathbf{x}(t) + \mathbf{u}
        \odot \mathbf{h}(t-1) + \mathbf{b} \right)
```

# Forward

    indrnncell(inp, state)
    indrnncell(inp)

## Arguments
- `inp`: The input to the indrnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the IndRNNCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct IndRNNCell{F, I, H, V, W, A} <: AbstractRecurrentCell
    activation::F
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer IndRNNCell

function IndRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int}, activation=relu;
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=true)
    weight_ih = init_kernel(hidden_size, input_size)
    if !independent_recurrence
        @warn"""\n
            IndRNNCell has independent_recurrence=true by default\n
        """
    end
    weight_hh = vec(init_recurrent_kernel(hidden_size))
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)
    return IndRNNCell(activation, weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (indrnn::IndRNNCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(indrnn, inp, 1 => size(indrnn.weight_ih, 2))
    activation = fast_act(indrnn.activation, inp)
    proj_ih = dense_proj(indrnn.weight_ih, inp, indrnn.bias_ih)
    proj_hh = dense_proj(indrnn.weight_hh, state, indrnn.bias_hh)
    state = activation.(indrnn.integration_fn(proj_ih, proj_hh))
    return state, state
end

function initialstates(indrnn::IndRNNCell)
    return zeros_like(indrnn.weight_hh, size(indrnn.weight_hh, 1))
end

function Base.show(io::IO, indrnn::IndRNNCell)
    print(io, "IndRNNCell(", size(indrnn.weight_ih, 2), " => ", size(indrnn.weight_ih, 1))
    print(io, ", ", indrnn.activation)
    print(io, ")")
end

@doc raw"""
    IndRNN(input_size, hidden_size, [activation];
        return_state = false, kwargs...)

Independently recurrent network [Li2018](@cite).
See [`IndRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: activation function. Default is `tanh`.

# Keyword arguments

- `return_state`: Option to return the last state together with the output.
  Default is `false`.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `true`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.


# Equations

```math
    \mathbf{h}(t) = \sigma\left( \mathbf{W}_{ih} \mathbf{x}(t) + \mathbf{u}
        \odot \mathbf{h}(t-1) + \mathbf{b} \right)
```

# Forward

    indrnn(inp, state)
    indrnn(inp)

## Arguments
- `inp`: The input to the indrnn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the IndRNN. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct IndRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand IndRNN

function IndRNN((input_size, hidden_size)::Pair{<:Int, <:Int}, activation=relu;
        return_state::Bool=false, kwargs...)
    cell = IndRNNCell(input_size => hidden_size, activation; kwargs...)
    return IndRNN{return_state, typeof(cell)}(cell)
end

function functor(rnn::IndRNN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> IndRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, indrnn::IndRNN)
    print(io, "IndRNN(", size(indrnn.cell.weight_ih, 2),
        " => ", size(indrnn.cell.weight_ih, 1))
    print(io, ", ", indrnn.cell.activation)
    print(io, ")")
end
