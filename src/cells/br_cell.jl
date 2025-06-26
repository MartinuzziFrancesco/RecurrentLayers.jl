#https://doi.org/10.1371/journal.pone.0252676
@doc raw"""
    BRCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform)


Bistable recurrent cell [Vecoven2021](@cite).
See [`BR`](@ref) for a layer that processes entire sequences.

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
    \mathbf{a}(t) &= 1 + \tanh\left( \mathbf{W}^{a}_{ih} \mathbf{x}(t) +
        \mathbf{w}^{a} \circ \mathbf{h}(t-1) + \mathbf{b}^{a} \right), \\
    \mathbf{c}(t) &= \sigma\left( \mathbf{W}^{c}_{ih} \mathbf{x}(t) +
        \mathbf{w}^{c} \circ \mathbf{h}(t-1) + \mathbf{b}^{c} \right), \\
    \mathbf{h}(t) &= \mathbf{c}(t) \circ \mathbf{h}(t-1) + \left(1 - \mathbf{c}(t)\right)
        \circ \tanh\left( \mathbf{W}^{h}_{ih} \mathbf{x}(t) + \mathbf{a}(t) \circ
        \mathbf{h}(t-1) + \mathbf{b}^{h} \right),
\end{aligned}
```

# Forward

    brcell(inp, state)
    brcell(inp)

## Arguments
- `inp`: The input to the brcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the BRCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct BRCell{I, H, V} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    b::V
end

@layer BRCell

function BRCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size * 3, input_size)
    Wh = init_recurrent_kernel(hidden_size * 2)
    b = create_bias(Wi, bias, size(Wi, 1))
    return BRCell(Wi, Wh, b)
end

function (br::BRCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(br, inp, 1 => size(br.Wi, 2))
    Wi, wh, b = br.Wi, vec(br.Wh), br.b
    gxs = chunk(Wi * inp .+ b, 3; dims=1)
    ws = chunk(wh, 2; dims=1)
    t_ones = eltype(Wi)(1.0)
    h1 = @. gxs[1] + ws[1] * state
    h2 = @. gxs[2] + ws[2] * state
    modulation_gate = @. t_ones + tanh_fast(h1)
    candidate_state = @. sigmoid_fast(h2)
    h3 = @. gxs[3] + modulation_gate * state
    new_state = @. candidate_state * state + (t_ones - candidate_state) * tanh_fast(h3)
    return new_state, new_state
end

function initialstates(br::BRCell)
    return zeros_like(br.Wh, size(br.Wh, 1) ÷ 2)
end

function Base.show(io::IO, br::BRCell)
    print(io, "BRCell(", size(br.Wi, 2), " => ", size(br.Wi, 1) ÷ 3)
    print(io, ")")
end

@doc raw"""
    BR(input_size, hidden_size;
        return_state = false, kwargs...)

Bistable recurrent network [Vecoven2021](@cite).
See [`BRCell`](@ref) for a layer that processes a single sequence.

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
    \mathbf{a}(t) &= 1 + \tanh\left( \mathbf{W}^{a}_{ih} \mathbf{x}(t) +
        \mathbf{w}^{a} \circ \mathbf{h}(t-1) + \mathbf{b}^{a} \right), \\
    \mathbf{c}(t) &= \sigma\left( \mathbf{W}^{c}_{ih} \mathbf{x}(t) +
        \mathbf{w}^{c} \circ \mathbf{h}(t-1) + \mathbf{b}^{c} \right), \\
    \mathbf{h}(t) &= \mathbf{c}(t) \circ \mathbf{h}(t-1) + \left(1 - \mathbf{c}(t)\right)
        \circ \tanh\left( \mathbf{W}^{h}_{ih} \mathbf{x}(t) + \mathbf{a}(t) \circ
        \mathbf{h}(t-1) + \mathbf{b}^{h} \right),
\end{aligned}
```

# Forward

    br(inp, state)
    br(inp)

## Arguments
- `inp`: The input to the br. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the BR. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct BR{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand BR

function BR((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = BRCell(input_size => hidden_size; kwargs...)
    return BR{return_state, typeof(cell)}(cell)
end

function functor(br::BR{S}) where {S}
    params = (cell=br.cell,)
    reconstruct = p -> BR{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, br::BR)
    print(
        io, "BR(", size(br.cell.Wi, 2), " => ", size(br.cell.Wi, 1) ÷ 3)
    print(io, ")")
end

@doc raw"""
    NBRCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform)


Recurrently neuromodulated bistable recurrent cell [Vecoven2021](@cite).
See [`NBR`](@ref) for a layer that processes entire sequences.

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
    \mathbf{a}(t) &= 1 + \tanh\left( \mathbf{W}^{a}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{a}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{a} \right), \\
    \mathbf{c}(t) &= \sigma\left( \mathbf{W}^{c}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{c}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{c} \right), \\
    \mathbf{h}(t) &= \mathbf{c}(t) \circ \mathbf{h}(t-1) + \left(1 -
        \mathbf{c}(t)\right) \circ \tanh\left( \mathbf{W}^{h}_{ih}
        \mathbf{x}(t) + \mathbf{a}(t) \circ \mathbf{h}(t-1) + \mathbf{b}^{h} \right),
\end{aligned}
```

# Forward

    nbrcell(inp, state)
    nbrcell(inp)

## Arguments
- `inp`: The input to the nbrcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the NBRCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct NBRCell{I, H, V} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    b::V
end

@layer NBRCell

function NBRCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size * 3, input_size)
    Wh = init_recurrent_kernel(hidden_size * 2, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    return NBRCell(Wi, Wh, b)
end

function (nbr::NBRCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(nbr, inp, 1 => size(nbr.Wi, 2))
    Wi, Wh, b = nbr.Wi, nbr.Wh, nbr.b
    gxs = chunk(Wi * inp .+ b, 3; dims=1)
    ghs = chunk(Wh * state, 2; dims=1)
    t_ones = eltype(Wi)(1.0)
    h1 = @. gxs[1] + gxs[1]
    h2 = @. gxs[2] + gxs[2]
    modulation_gate = @. t_ones + tanh_fast(h1)
    candidate_state = @. sigmoid_fast(h2)
    h3 = @. gxs[3] + modulation_gate * state
    new_state = @. candidate_state * state + (t_ones - candidate_state) * tanh_fast(h3)
    return new_state, new_state
end

function Base.show(io::IO, nbr::NBRCell)
    print(io, "NBRCell(", size(nbr.Wi, 2), " => ", size(nbr.Wi, 1) ÷ 3)
    print(io, ")")
end

@doc raw"""
    NBR(input_size, hidden_size;
        return_state = false, kwargs...)

Recurrently neuromodulated bistable recurrent cell [Vecoven2021](@cite).
See [`NBRCell`](@ref) for a layer that processes a single sequence.

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
    \mathbf{a}(t) &= 1 + \tanh\left( \mathbf{W}^{a}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{a}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{a} \right), \\
    \mathbf{c}(t) &= \sigma\left( \mathbf{W}^{c}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{c}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{c} \right), \\
    \mathbf{h}(t) &= \mathbf{c}(t) \circ \mathbf{h}(t-1) + \left(1 -
        \mathbf{c}(t)\right) \circ \tanh\left( \mathbf{W}^{h}_{ih}
        \mathbf{x}(t) + \mathbf{a}(t) \circ \mathbf{h}(t-1) + \mathbf{b}^{h} \right),
\end{aligned}
```

# Forward

    nbr(inp, state)
    nbr(inp)

## Arguments
- `inp`: The input to the nbr. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the NBR. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct NBR{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand NBR

function NBR((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = NBRCell(input_size => hidden_size; kwargs...)
    return NBR{return_state, typeof(cell)}(cell)
end

function functor(nbr::NBR{S}) where {S}
    params = (cell=nbr.cell,)
    reconstruct = p -> NBR{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, nbr::NBR)
    print(
        io, "NBR(", size(nbr.cell.Wi, 2), " => ", size(nbr.cell.Wi, 1) ÷ 3)
    print(io, ")")
end
