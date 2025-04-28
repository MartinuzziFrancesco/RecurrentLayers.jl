#https://doi.org/10.1371/journal.pone.0252676
@doc raw"""
    BRCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform)


[Bistable recurrent cell](https://doi.org/10.1371/journal.pone.0252676).
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
    \mathbf{h}_t &= \mathbf{c}_t \circ \mathbf{h}_{t-1} + (1 - \mathbf{c}_t)
        \circ \tanh\left(\mathbf{U}_x \mathbf{x}_t + \mathbf{a}_t \circ
        \mathbf{h}_{t-1}\right), \\
    \mathbf{a}_t &= 1 + \tanh\left(\mathbf{U}_a \mathbf{x}_t +
        \mathbf{w}_a \circ \mathbf{h}_{t-1}\right), \\
    \mathbf{c}_t &= \sigma\left(\mathbf{U}_c \mathbf{x}_t + \mathbf{w}_c \circ
        \mathbf{h}_{t-1}\right).
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
    Wh = init_recurrent_kernel(hidden_size * 3)
    b = create_bias(Wi, bias, size(Wi, 1))
    return BRCell(Wi, Wh, b)
end

function (br::BRCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(br, inp, 1 => size(br.Wi, 2))
    Wi, wh, b = br.Wi, vec(br.Wh), br.b
    gxs = chunk(Wi * inp .+ b, 3; dims=1)
    ws = chunk(wh, 3; dims=1)
    ones = eltype(Wi)(1.0)
    h1 = @. gxs[1] + ws[1] * state
    h2 = @. gxs[2] + ws[2] * state
    h3 = @. gxs[3] + ws[3] * state
    modulation_gate = @. ones + tanh_fast(h1)
    candidate_state = @. sigmoid_fast(h2)
    new_state = @. candidate_state * state + (ones - candidate_state) * tanh_fast(h3)
    return new_state, new_state
end

function initialstates(br::BRCell)
    return zeros_like(br.Wh, size(br.Wh, 1) ÷ 3)
end

function Base.show(io::IO, br::BRCell)
    print(io, "BRCell(", size(br.Wi, 2), " => ", size(br.Wi, 1) ÷ 3)
    print(io, ")")
end

@doc raw"""
    BR(input_size, hidden_size;
        return_state = false, kwargs...)

[Bistable recurrent network](https://doi.org/10.1371/journal.pone.0252676).
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
    \mathbf{h}_t &= \mathbf{c}_t \circ \mathbf{h}_{t-1} + (1 - \mathbf{c}_t)
        \circ \tanh\left(\mathbf{U}_x \mathbf{x}_t + \mathbf{a}_t \circ
        \mathbf{h}_{t-1}\right), \\
    \mathbf{a}_t &= 1 + \tanh\left(\mathbf{U}_a \mathbf{x}_t +
        \mathbf{w}_a \circ \mathbf{h}_{t-1}\right), \\
    \mathbf{c}_t &= \sigma\left(\mathbf{U}_c \mathbf{x}_t + \mathbf{w}_c \circ
        \mathbf{h}_{t-1}\right).
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
    reconstruct = p -> AntisymmetricRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, br::BR)
    print(
        io, "BR(", size(br.cell.Wi, 2), " => ", size(br.cell.Wi, 1) ÷ 3)
    print(io, ")")
end
