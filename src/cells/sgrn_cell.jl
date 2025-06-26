#https://doi.org/10.1049/gtd2.12056
@doc raw"""
    SGRNCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform)


Simple gated recurrent network [Zu2020](@cite).
See [`SGRN`](@ref) for a layer that processes entire sequences.

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
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}_{ih} \mathbf{x}(t) +
        \mathbf{W}_{hh} \mathbf{h}(t-1) + \mathbf{b} \right) \\
    \mathbf{i}(t) &= 1 - \mathbf{f}(t) \\
    \mathbf{h}(t) &= \tanh\left( \mathbf{i}(t) \circ \left(
        \mathbf{W}_{ih} \mathbf{x}(t) \right) + \mathbf{f}(t) \circ
        \mathbf{h}(t-1) \right)
\end{aligned}
```

# Forward

    sgrncell(inp, state)
    sgrncell(inp)

## Arguments
- `inp`: The input to the sgrncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the SGRNCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct SGRNCell{I, H, V} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    b::V
end

@layer SGRNCell

function SGRNCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    return SGRNCell(Wi, Wh, b)
end

function (sgrn::SGRNCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(sgrn, inp, 1 => size(sgrn.Wi, 2))
    Wi, Wh, b = sgrn.Wi, sgrn.Wh, sgrn.b
    xs = Wi * inp
    hs = Wh * state .+ b
    forget_gate = @. sigmoid_fast(xs + hs)
    input_gate = eltype(Wi)(1.0) .- forget_gate
    new_state = @. tanh_fast(input_gate * xs + forget_gate * state)
    return new_state, new_state
end

function Base.show(io::IO, sgrn::SGRNCell)
    print(io, "SGRNCell(", size(sgrn.Wi, 2), " => ", size(sgrn.Wi, 1))
    print(io, ")")
end

@doc raw"""
    SGRN(input_size, hidden_size;
        return_state = false, kwargs...)

Simple gated recurrent network [Zu2020](@cite).
See [`SGRNCell`](@ref) for a layer that processes a single sequence.

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
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}_{ih} \mathbf{x}(t) +
        \mathbf{W}_{hh} \mathbf{h}(t-1) + \mathbf{b} \right) \\
    \mathbf{i}(t) &= 1 - \mathbf{f}(t) \\
    \mathbf{h}(t) &= \tanh\left( \mathbf{i}(t) \circ \left(
        \mathbf{W}_{ih} \mathbf{x}(t) \right) + \mathbf{f}(t) \circ
        \mathbf{h}(t-1) \right)
\end{aligned}
```

# Forward

    sgrn(inp, state)
    sgrn(inp)

## Arguments
- `inp`: The input to the sgrn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the SGRN. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct SGRN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand SGRN

function SGRN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = SGRNCell(input_size => hidden_size; kwargs...)
    return SGRN{return_state, typeof(cell)}(cell)
end

function functor(sgrn::SGRN{S}) where {S}
    params = (cell=sgrn.cell,)
    reconstruct = p -> AntisymmetricRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, sgrn::SGRN)
    print(
        io, "SGRN(", size(sgrn.cell.Wi, 2), " => ", size(sgrn.cell.Wi, 1))
    print(io, ")")
end
