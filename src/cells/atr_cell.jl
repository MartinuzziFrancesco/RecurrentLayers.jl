#https://arxiv.org/abs/1810.12546
@doc raw"""
    ATRCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform)


[Addition-subtraction twin-gated recurrent cell](https://arxiv.org/abs/1810.12546).
See [`ATR`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{p}(t) &= \mathbf{W}_{ih} \mathbf{x}(t) + \mathbf{b}, \\
    \mathbf{q}(t) &= \mathbf{W}_{hh} \mathbf{h}(t-1), \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{p}(t) + \mathbf{q}(t) \right), \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{p}(t) - \mathbf{q}(t) \right), \\
    \mathbf{h}(t) &= \mathbf{i}(t) \circ \mathbf{p}(t) + \mathbf{f}(t) \circ
        \mathbf{h}(t-1).
\end{aligned}

```

# Forward

    atrcell(inp, state)
    atrcell(inp)

## Arguments
- `inp`: The input to the atrcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the ATRCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct ATRCell{I, H, V} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    b::V
end

@layer ATRCell

function ATRCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    return ATRCell(Wi, Wh, b)
end

function (atr::ATRCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(atr, inp, 1 => size(atr.Wi, 2))
    Wi, Wh, b = atr.Wi, atr.Wh, atr.b
    pt = Wi * inp .+ b
    qt = Wh * state
    it = @. sigmoid_fast(pt + qt)
    ft = @. sigmoid_fast(pt - qt)
    new_state = @. it * pt + ft * state
    return new_state, new_state
end

function Base.show(io::IO, atr::ATRCell)
    print(io, "ATRCell(", size(atr.Wi, 2), " => ", size(atr.Wi, 1))
    print(io, ")")
end

@doc raw"""
    ATR(input_size, hidden_size;
        return_state = false, kwargs...)

[Addition-subtraction twin-gated recurrent cell](https://arxiv.org/abs/1810.12546).
See [`ATRCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`

# Equations

```math
\begin{aligned}
    \mathbf{p}(t) &= \mathbf{W}_{ih} \mathbf{x}(t) + \mathbf{b}, \\
    \mathbf{q}(t) &= \mathbf{W}_{hh} \mathbf{h}(t-1), \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{p}(t) + \mathbf{q}(t) \right), \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{p}(t) - \mathbf{q}(t) \right), \\
    \mathbf{h}(t) &= \mathbf{i}(t) \circ \mathbf{p}(t) + \mathbf{f}(t) \circ
        \mathbf{h}(t-1).
\end{aligned}

```

# Forward

    atr(inp, state)
    atr(inp)

## Arguments
- `inp`: The input to the atr. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the ATR. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct ATR{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand ATR

function ATR((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = ATRCell(input_size => hidden_size; kwargs...)
    return ATR{return_state, typeof(cell)}(cell)
end

function functor(atr::ATR{S}) where {S}
    params = (cell=atr.cell,)
    reconstruct = p -> ATR{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, atr::ATR)
    print(
        io, "ATR(", size(atr.cell.Wi, 2), " => ", size(atr.cell.Wi, 1))
    print(io, ")")
end
