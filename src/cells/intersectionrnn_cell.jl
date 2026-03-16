@doc raw"""
    IntersectionRNNCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Minimal gated unit [Collins2016](@cite).
See [`IntersectionRNN`](@ref) for a layer that processes entire sequences.

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
    \mathbf{y}^{in}(t) &= s_1\left( \mathbf{W}_{hh}^y \, \mathbf{h}(t-1) +
        \mathbf{W}_{xh}^y \, \mathbf{x}(t) + \mathbf{b}^y \right), \\
    \mathbf{h}^{in}(t) &= s_2\left( \mathbf{W}_{hh}^h \, \mathbf{h}(t-1) +
        \mathbf{W}_{xh}^h \, \mathbf{x}(t) + \mathbf{b}^h \right), \\
    \mathbf{g}^y(t) &= \sigma\left( \mathbf{W}_{hh}^{g^y} \, \mathbf{h}(t-1)
        + \mathbf{W}_{xh}^{g^y} \, \mathbf{x}(t) + \mathbf{b}^{g^y}  \right), \\
    \mathbf{g}^h(t) &= \sigma\left( \mathbf{W}_{hh}^{g^h} \, \mathbf{h}(t-1) +
        \mathbf{W}_{xh}^{g^h} \, \mathbf{x}(t) + \mathbf{b}^{g^h}  \right), \\
    \mathbf{y}(t) &= \mathbf{g}^y(t) \circ \mathbf{x}(t) + \left( 1 -
        \mathbf{g}^y(t) \right) \circ \mathbf{y}^{in}(t), \\
    \mathbf{h}(t) &= \mathbf{g}^h(t) \circ \mathbf{h}(t-1) + \left( 1 -
        \mathbf{g}^h(t) \right) \circ \mathbf{h}^{in}(t).
\end{aligned}
```

# Forward

    intersectionrnncell(inp, state)
    intersectionrnncell(inp)

## Arguments
- `inp`: The input to the intersectionrnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the IntersectionRNNCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct IntersectionRNNCell{I, H, V, W, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

function IntersectionRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition, independent_recurrence::Bool=true)
    weight_ih = init_kernel(hidden_size * 4, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 4)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)

    return IntersectionRNNCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (irnn::IntersectionRNNCell)(inp::AbstractVecOrMat, state)
    _size_check(irnn, inp, 1 => size(irnn.weight_ih, 2))
    proj_ih = dense_proj(irnn.weight_ih, inp, irnn.bias_ih)
    proj_hh = dense_proj(irnn.weight_hh, state, irnn.bias_hh)
    gxs = chunk(proj_ih, 4; dims=1)
    ghs = chunk(proj_hh, 4; dims=1)
    yin = relu.(irnn.integration_fn(gxs[1], ghs[1]))
    hin = tanh_fast.(irnn.integration_fn(gxs[2], ghs[2]))
    gy = sigmoid_fast.(irnn.integration_fn(gxs[3], ghs[3]))
    gh = sigmoid_fast.(irnn.integration_fn(gxs[4], ghs[4]))
    new_inp = gy .* inp .+ (1 .- gh) .* yin
    new_state = gh .* state .+ (1 .- gh) .* hin

    return new_inp, new_state ## double check this
end

function initialstates(irnn::IntersectionRNNCell)
    return zeros_like(irnn.weight_hh, size(irnn.weight_hh, 1) ÷ 4)
end

function Base.show(io::IO, irnn::IntersectionRNNCell)
    print(io, "IntersectionRNNCell(", size(irnn.weight_ih, 2),
        " => ", size(irnn.weight_ih, 1) ÷ 4, ")")
end

@doc raw"""
    IntersectionRNN(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Minimal gated unit [Collins2016](@cite).
See [`IntersectionRNNCell`](@ref) for a layer that processes a single sequence.

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
    \mathbf{y}^{in}(t) &= s_1\left( \mathbf{W}_{hh}^y \, \mathbf{h}(t-1) +
        \mathbf{W}_{xh}^y \, \mathbf{x}(t) + \mathbf{b}^y \right), \\
    \mathbf{h}^{in}(t) &= s_2\left( \mathbf{W}_{hh}^h \, \mathbf{h}(t-1) +
        \mathbf{W}_{xh}^h \, \mathbf{x}(t) + \mathbf{b}^h \right), \\
    \mathbf{g}^y(t) &= \sigma\left( \mathbf{W}_{hh}^{g^y} \, \mathbf{h}(t-1)
        + \mathbf{W}_{xh}^{g^y} \, \mathbf{x}(t) + \mathbf{b}^{g^y}  \right), \\
    \mathbf{g}^h(t) &= \sigma\left( \mathbf{W}_{hh}^{g^h} \, \mathbf{h}(t-1) +
        \mathbf{W}_{xh}^{g^h} \, \mathbf{x}(t) + \mathbf{b}^{g^h}  \right), \\
    \mathbf{y}(t) &= \mathbf{g}^y(t) \circ \mathbf{x}(t) + \left( 1 -
        \mathbf{g}^y(t) \right) \circ \mathbf{y}^{in}(t), \\
    \mathbf{h}(t) &= \mathbf{g}^h(t) \circ \mathbf{h}(t-1) + \left( 1 -
        \mathbf{g}^h(t) \right) \circ \mathbf{h}^{in}(t).
\end{aligned}
```

# Forward

    intersectionrnn(inp, state)
    intersectionrnn(inp)

## Arguments
- `inp`: The input to the intersectionrnn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the IntersectionRNN. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct IntersectionRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand IntersectionRNN

function IntersectionRNN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = IntersectionRNNCell(input_size => hidden_size; kwargs...)
    return IntersectionRNN{return_state, typeof(cell)}(cell)
end

function functor(rnn::IntersectionRNN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> IntersectionRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, irnn::IntersectionRNN)
    print(io, "IntersectionRNN(", size(irnn.cell.weight_ih, 2),
        " => ", size(irnn.cell.bias_ih, 1) ÷ 4)
    print(io, ")")
end
