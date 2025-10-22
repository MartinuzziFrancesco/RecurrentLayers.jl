#https://arxiv.org/pdf/1603.09420
@doc raw"""
    MGUCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Minimal gated unit [Zhou2016](@cite).
See [`MGU`](@ref) for a layer that processes entire sequences.

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
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{f} \right), \\
    \tilde{\mathbf{h}}(t) &= \tanh\left( \mathbf{W}^{h}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{h}_{hh} \left( \mathbf{f}(t) \odot \mathbf{h}(t-1) \right) +
        \mathbf{b}^{h} \right), \\
    \mathbf{h}(t) &= \left(1 - \mathbf{f}(t)\right) \odot \mathbf{h}(t-1) +
        \mathbf{f}(t) \odot \tilde{\mathbf{h}}(t)
\end{aligned}
```

# Forward

    mgucell(inp, state)
    mgucell(inp)

## Arguments
- `inp`: The input to the mgucell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the MGUCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct MGUCell{I, H, V, W, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer MGUCell

function MGUCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(2 * hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 2)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)
    return MGUCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (mgu::MGUCell)(inp::AbstractVecOrMat, state)
    _size_check(mgu, inp, 1 => size(mgu.weight_ih, 2))
    proj_ih = dense_proj(mgu.weight_ih, inp, mgu.bias_ih)
    gxs = chunk(proj_ih, 2; dims=1)
    ghs = chunk(mgu.weight_hh, 2; dims=1)
    bhs = chunk(mgu.bias_hh, 2; dims=1)
    t_ones = eltype(mgu.weight_ih)(1.0f0)
    proj_hh_1 = dense_proj(ghs[1], state, bhs[1])
    forget_gate = sigmoid_fast.(mgu.integration_fn(gxs[1], proj_hh_1))
    proj_hh_2 = dense_proj(ghs[2], forget_gate .* state, bhs[2])
    candidate_state = tanh_fast.(mgu.integration_fn(gxs[2], proj_hh_2))
    new_state = @. forget_gate * state + (t_ones - forget_gate) * candidate_state
    return new_state, new_state
end

function initialstates(mgu::MGUCell)
    return zeros_like(mgu.weight_hh, size(mgu.weight_hh, 1) ÷ 2)
end

function Base.show(io::IO, mgu::MGUCell)
    print(io, "MGUCell(", size(mgu.weight_ih, 2), " => ", size(mgu.weight_ih, 1) ÷ 2, ")")
end

@doc raw"""
    MGU(input_size => hidden_size;
        return_state = false, kwargs...)

Minimal gated unit network [Zhou2016](@cite).
See [`MGUCell`](@ref) for a layer that processes a single sequence.

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
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{f} \right), \\
    \tilde{\mathbf{h}}(t) &= \tanh\left( \mathbf{W}^{h}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{h}_{hh} \left( \mathbf{f}(t) \odot \mathbf{h}(t-1) \right) +
        \mathbf{b}^{h} \right), \\
    \mathbf{h}(t) &= \left(1 - \mathbf{f}(t)\right) \odot \mathbf{h}(t-1) +
        \mathbf{f}(t) \odot \tilde{\mathbf{h}}(t)
\end{aligned}
```

# Forward

    mgu(inp, state)
    mgu(inp)

## Arguments
- `inp`: The input to the mgu. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the MGU. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct MGU{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand MGU

function MGU((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = MGUCell(input_size => hidden_size; kwargs...)
    return MGU{return_state, typeof(cell)}(cell)
end

function functor(rnn::MGU{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> MGU{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, mgu::MGU)
    print(io, "MGU(", size(mgu.cell.bias_ih, 2), " => ", size(mgu.cell.bias_ih, 1) ÷ 2)
    print(io, ")")
end
