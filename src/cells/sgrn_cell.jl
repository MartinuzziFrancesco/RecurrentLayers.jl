#https://doi.org/10.1049/gtd2.12056
@doc raw"""
    SGRNCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)


Simple gated recurrent network [Zu2020](@cite).
See [`SGRN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

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
struct SGRNCell{I, H, V, W, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer SGRNCell

function SGRNCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)
    return SGRNCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (sgrn::SGRNCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(sgrn, inp, 1 => size(sgrn.weight_ih, 2))
    proj_ih = dense_proj(sgrn.weight_ih, inp, sgrn.bias_ih)
    proj_hh = dense_proj(sgrn.weight_hh, state, sgrn.bias_hh)
    forget_gate = sigmoid_fast.(sgrn.integration_fn(proj_ih, proj_hh))
    input_gate = eltype(sgrn.weight_ih)(1.0) .- forget_gate
    new_state = @. tanh_fast(input_gate * proj_ih + forget_gate * state)
    return new_state, new_state
end

function initialstates(sgrn::SGRNCell)
    return zeros_like(sgrn.weight_hh, size(sgrn.weight_hh, 1))
end

function Base.show(io::IO, sgrn::SGRNCell)
    print(io, "SGRNCell(", size(sgrn.weight_ih, 2), " => ", size(sgrn.weight_ih, 1))
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
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.

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
        io, "SGRN(", size(sgrn.cell.weight_ih, 2), " => ", size(sgrn.cell.weight_ih, 1))
    print(io, ")")
end
