#https://arxiv.org/abs/1810.12546
@doc raw"""
    ATRCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        independent_recurrence = false, integration_mode = :addition,
        bias = true, recurrent_bias = true,)


Addition-subtraction twin-gated recurrent cell [Zhang2018](@cite).
See [`ATR`](@ref) for a layer that processes entire sequences.

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
struct ATRCell{I, H, V, W, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer ATRCell

function ATRCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition, independent_recurrence::Bool=false)
    weight_ih = init_kernel(hidden_size, input_size)
    if independent_recurrence
        weight_hh = vec(init_recurrent_kernel(hidden_size))
    else
        weight_hh = init_recurrent_kernel(hidden_size, hidden_size)
    end
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    if integration_mode == :addition
        integration_fn = add_projections
    elseif integration_mode == :multiplicative_integration
        integration_fn = mul_projections
        @warn """\n
            multiplicative_integration removes the benefits of this architecture.
            Defaulting to :addition
            \n
            """
    else
        throw(ArgumentError(
            "integration_mode must be :addition or :multiplicative_integration; got $integration_mode"
        ))
    end
    return ATRCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (atr::ATRCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(atr, inp, 1 => size(atr.weight_ih, 2))
    proj_ih = dense_proj(atr.weight_ih, inp, atr.bias_ih) #pt
    proj_hh = dense_proj(atr.weight_hh, state, atr.bias_hh) #qt
    it = @. sigmoid_fast(proj_ih + proj_hh)
    ft = @. sigmoid_fast(proj_ih - proj_hh)
    new_state = @. it * proj_ih + ft * state
    return new_state, new_state
end

function initialstates(atr::ATRCell)
    return zeros_like(atr.weight_hh, size(atr.weight_hh, 1))
end

function Base.show(io::IO, atr::ATRCell)
    print(io, "ATRCell(", size(atr.weight_ih, 2), " => ", size(atr.weight_ih, 1))
    print(io, ")")
end

@doc raw"""
    ATR(input_size, hidden_size;
        return_state = false, kwargs...)

Addition-subtraction twin-gated recurrent cell [Zhang2018](@cite).
See [`ATRCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

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
        io, "ATR(", size(atr.cell.weight_ih, 2), " => ", size(atr.cell.weight_ih, 1))
    print(io, ")")
end
