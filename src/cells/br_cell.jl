#https://doi.org/10.1371/journal.pone.0252676
@doc raw"""
    BRCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)


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
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: hard-coded to `true` in this architecture. For the
  architecture without independent recurrence plese refer to [`NBRCell`](@ref)
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.

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
struct BRCell{I, H, V, W, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer BRCell

function BRCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition, independent_recurrence::Bool=true)
    weight_ih = init_kernel(hidden_size * 3, input_size)
    weight_hh = init_recurrent_kernel(hidden_size * 2)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    if !independent_recurrence
        @warn "independent_recurrence defaults to true in BRCell"
    end
    if integration_mode == :addition
        integration_fn = add_projections
    elseif integration_mode == :multiplicative_integration
        integration_fn = mul_projections
    else
        throw(ArgumentError(
            "integration_mode must be :addition or :multiplicative_integration; got $integration_mode"
        ))
    end
    return BRCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (br::BRCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(br, inp, 1 => size(br.weight_ih, 2))
    proj_ih = dense_proj(br.weight_ih, inp, br.bias_ih)
    gxs = chunk(proj_ih, 3; dims=1)
    whs = chunk(br.weight_hh, 2; dims=1)
    bhs = chunk(br.bias_hh, 2; dims=1)
    proj_ih_1 = dense_proj(whs[1], state, bhs[1])
    proj_ih_2 = dense_proj(whs[2], state, bhs[2])
    t_ones = eltype(br.weight_ih)(1.0)
    h1 = br.integration_fn(gxs[1], proj_ih_1)
    h2 = br.integration_fn(gxs[2], proj_ih_2)
    modulation_gate = @. t_ones + tanh_fast(h1)
    candidate_state = @. sigmoid_fast(h2)
    h3 = @. gxs[3] + modulation_gate * state
    new_state = @. candidate_state * state + (t_ones - candidate_state) * tanh_fast(h3)
    return new_state, new_state
end

function initialstates(br::BRCell)
    return zeros_like(br.weight_hh, size(br.weight_hh, 1) ÷ 2)
end

function Base.show(io::IO, br::BRCell)
    print(io, "BRCell(", size(br.weight_ih, 2), " => ", size(br.weight_ih, 1) ÷ 3)
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
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: hard-coded to `true` in this architecture. For the
  architecture without independent recurrence plese refer to [`NBR`](@ref)
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.

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
        io, "BR(", size(br.cell.weight_ih, 2), " => ", size(br.cell.weight_ih, 1) ÷ 3)
    print(io, ")")
end

@doc raw"""
    NBRCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)


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
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: hard-coded to `false` in this architecture. For the
  architecture with independent recurrence plese refer to [`BRCell`](@ref)
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.

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
struct NBRCell{I, H, V, W, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer NBRCell

function NBRCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition, independent_recurrence::Bool=false)
    weight_ih = init_kernel(hidden_size * 3, input_size)
    weight_hh = init_recurrent_kernel(hidden_size * 2, hidden_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    if independent_recurrence
        @warn "independent_recurrence defaults to false in NBRCell"
    end
    if integration_mode == :addition
        integration_fn = add_projections
    elseif integration_mode == :multiplicative_integration
        integration_fn = mul_projections
    else
        throw(ArgumentError(
            "integration_mode must be :addition or :multiplicative_integration; got $integration_mode"
        ))
    end
    return NBRCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (nbr::NBRCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(nbr, inp, 1 => size(nbr.weight_ih, 2))
    proj_ih = dense_proj(nbr.weight_ih, inp, nbr.bias_ih)
    proj_hh = dense_proj(nbr.weight_hh, state, nbr.bias_hh)
    gxs = chunk(proj_ih, 3; dims=1)
    ghs = chunk(proj_hh, 2; dims=1)
    t_ones = eltype(nbr.weight_ih)(1.0)
    h1 = nbr.integration_fn(gxs[1], gxs[1])
    h2 = nbr.integration_fn(gxs[2], gxs[2])
    modulation_gate = @. t_ones + tanh_fast(h1)
    candidate_state = @. sigmoid_fast(h2)
    h3 = @. gxs[3] + modulation_gate * state
    new_state = @. candidate_state * state + (t_ones - candidate_state) * tanh_fast(h3)
    return new_state, new_state
end

function Base.show(io::IO, nbr::NBRCell)
    print(io, "NBRCell(", size(nbr.weight_ih, 2), " => ", size(nbr.weight_ih, 1) ÷ 3)
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
- `bias`: include a bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: hard-coded to `false` in this architecture. For the
  architecture with independent recurrence plese refer to [`BR`](@ref)
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.


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
        io, "NBR(", size(nbr.cell.weight_ih, 2), " => ", size(nbr.cell.weight_ih, 1) ÷ 3)
    print(io, ")")
end
