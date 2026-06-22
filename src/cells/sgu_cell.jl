@doc raw"""
    SGUCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Simple gated unit [Gao2016](@cite).
See [`SGU`](@ref) for a layer that processes entire sequences.

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
    \mathbf{x}_g(t) &= \mathbf{W}_{xg} \mathbf{x}(t), \\
    \mathbf{z}_g(t) &= \tanh\left(\mathbf{W}_{zg}
        \left(\mathbf{x}_g(t) \circ \mathbf{h}(t-1)\right) +
        \mathbf{b}_{zg}\right), \\
    \mathbf{z}_{out}(t) &= \operatorname{softplus}\left(
        \mathbf{z}_g(t) \circ \mathbf{h}(t-1)\right), \\
    \mathbf{z}_t &= \operatorname{hard sigmoid}\left(
        \mathbf{W}_{xz} \mathbf{x}(t) + \mathbf{W}_{hz} \mathbf{h}(t-1) +
        \mathbf{b}_z\right), \\
    \mathbf{h}(t) &= \left(1 - \mathbf{z}_t\right) \circ \mathbf{h}(t-1) +
        \mathbf{z}_t \circ \mathbf{z}_{out}(t).
\end{aligned}
```

# Forward

    sgucell(inp, state)
    sgucell(inp)

## Arguments
- `inp`: The input to the sgucell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the SGUCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct SGUCell{I, H, V, W, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer SGUCell

function SGUCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(2 * hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 2)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)
    return SGUCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (sgucell::SGUCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(sgucell, inp, 1 => size(sgucell.weight_ih, 2))
    proj_ih = dense_proj(sgucell.weight_ih, inp, sgucell.bias_ih)
    gxs = chunk(proj_ih, 2; dims=1)
    whs = chunk(sgucell.weight_hh, 2; dims=1)
    bhs = _chunked_bias(sgucell.bias_hh, 2)
    zg = tanh_fast.(dense_proj(whs[1], gxs[1] .* state, bhs[1]))
    zout = softplus.(zg .* state)
    zt_c1 = dense_proj(whs[2], state, bhs[2])
    zt = hardsigmoid.(sgucell.integration_fn(gxs[2], zt_c1))
    new_state = @. (1 - zt) * state + zt * zout
    return new_state, new_state
end

function initialstates(sgucell::SGUCell)
    state = zeros_like(sgucell.weight_hh, size(sgucell.weight_hh, 1) ÷ 2)
    return state
end

function Base.show(io::IO, sgucell::SGUCell)
    print(io, "SGUCell(", size(sgucell.weight_ih, 2),
        " => ", size(sgucell.weight_ih, 1) ÷ 2, ")")
end

@doc raw"""
    SGU(input_size => hidden_size;
        return_state = false, kwargs...)

Simple gated unit network [Gao2016](@cite).
See [`SGUCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

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
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.

# Equations

```math
\begin{aligned}
    \mathbf{x}_g(t) &= \mathbf{W}_{xg} \mathbf{x}(t), \\
    \mathbf{z}_g(t) &= \tanh\left(\mathbf{W}_{zg}
        \left(\mathbf{x}_g(t) \circ \mathbf{h}(t-1)\right) +
        \mathbf{b}_{zg}\right), \\
    \mathbf{z}_{out}(t) &= \operatorname{softplus}\left(
        \mathbf{z}_g(t) \circ \mathbf{h}(t-1)\right), \\
    \mathbf{z}_t &= \operatorname{hard sigmoid}\left(
        \mathbf{W}_{xz} \mathbf{x}(t) + \mathbf{W}_{hz} \mathbf{h}(t-1) +
        \mathbf{b}_z\right), \\
    \mathbf{h}(t) &= \left(1 - \mathbf{z}_t\right) \circ \mathbf{h}(t-1) +
        \mathbf{z}_t \circ \mathbf{z}_{out}(t).
\end{aligned}
```

# Forward

    sgu(inp, state)
    sgu(inp)

## Arguments
- `inp`: The input to the sgu. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the SGU. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden states `new_states` and
  the last state of the iteration.
"""
struct SGU{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand SGU

function SGU((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = SGUCell(input_size => hidden_size; kwargs...)
    return SGU{return_state, typeof(cell)}(cell)
end

function functor(rnn::SGU{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> SGU{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, sgu::SGU)
    print(io, "SGU(", size(sgu.cell.weight_ih, 2),
        " => ", size(sgu.cell.weight_ih, 1) ÷ 2)
    print(io, ")")
end
