#https://arxiv.org/pdf/1803.04831

@doc raw"""
    IndRNNCell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)


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
- `bias`: include a bias or not. Default is `true`.

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
struct IndRNNCell{F, I, H, V} <: AbstractRecurrentCell
    activation::F
    Wi::I
    Wh::H
    b::V
end

@layer IndRNNCell

function IndRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int}, activation=relu;
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    return IndRNNCell(activation, Wi, Wh, b)
end

function (indrnn::IndRNNCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(indrnn, inp, 1 => size(indrnn.Wi, 2))
    activation = fast_act(indrnn.activation, inp)
    state = activation.(indrnn.Wi * inp .+ vec(indrnn.Wh) .* state .+ indrnn.b)
    return state, state
end

function initialstates(indrnn::IndRNNCell)
    return zeros_like(indrnn.Wh, size(indrnn.Wh, 1))
end

function Base.show(io::IO, indrnn::IndRNNCell)
    print(io, "IndRNNCell(", size(indrnn.Wi, 2), " => ", size(indrnn.Wi, 1))
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
- `bias`: include a bias or not. Default is `true`.

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
    print(io, "IndRNN(", size(indrnn.cell.Wi, 2), " => ", size(indrnn.cell.Wi, 1))
    print(io, ", ", indrnn.cell.activation)
    print(io, ")")
end
