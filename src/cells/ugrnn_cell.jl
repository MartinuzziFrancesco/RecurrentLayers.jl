#https://arxiv.org/pdf/1611.09913

@doc raw"""
    UGRNNCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Update gate recurrent unit [Collins2017](@cite).
See [`UGRNN`](@ref) for a layer that processes entire sequences.

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
    \mathbf{c}(t) &= s\left( \mathbf{W}_{hh}^c \, \mathbf{h}(t-1) + \mathbf{W}_{xh}^c \,
        \mathbf{x}(t) + \mathbf{b}^c \right), \\
    \mathbf{g}(t) &= \sigma\left( \mathbf{W}_{hh}^g \, \mathbf{h}(t-1) +
        \mathbf{W}_{xh}^g \, \mathbf{x}(t) + \mathbf{b}^g  \right), \\
    \mathbf{h}(t) &= \mathbf{g}(t) \circ \mathbf{h}(t-1) + \left( 1 -
        \mathbf{g}(t) \right) \circ \mathbf{c}(t).
\end{aligned}
```

# Forward

    ugrnncell(inp, state)
    ugrnncell(inp)

## Arguments
- `inp`: The input to the ugrnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the UGRNNCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct UGRNNCell{I, H, V, W, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer UGRNNCell

function UGRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(2 * hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 2)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)
    return UGRNNCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (ugrnn::UGRNNCell)(inp::AbstractVecOrMat, state)
    _size_check(ugrnn, inp, 1 => size(ugrnn.weight_ih, 2))
    proj_ih = dense_proj(ugrnn.weight_ih, inp, ugrnn.bias_ih)
    proj_hh = dense_proj(ugrnn.weight_hh, state, ugrnn.bias_hh)
    merged_proj = ugrnn.integration_fn(proj_ih, proj_hh)
    gates = chunk(merged_proj, 2; dims=1)
    candidate_state = @. tanh_fast(gates[1])
    update_gate = sigmoid_fast.(gates[2])
    new_state = @. update_gate * state + (1 - update_gate) * candidate_state
    return new_state, new_state
end

function initialstates(ugrnn::UGRNNCell)
    return zeros_like(ugrnn.weight_hh, size(ugrnn.weight_hh, 1) ÷ 2)
end

function Base.show(io::IO, ugrnn::UGRNNCell)
    print(io, "UGRNNCell(", size(ugrnn.weight_ih, 2),
        " => ", size(ugrnn.weight_ih, 1) ÷ 2, ")")
end

@doc raw"""
    UGRNN(input_size => hidden_size;
        return_state = false, kwargs...)

Update gate recurrent neural network [Collins2017](@cite).
See [`UGRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer

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
    \mathbf{c}(t) &= s\left( \mathbf{W}_{hh}^c \, \mathbf{h}(t-1) + \mathbf{W}_{xh}^c \,
        \mathbf{x}(t) + \mathbf{b}^c \right), \\
    \mathbf{g}(t) &= \sigma\left( \mathbf{W}_{hh}^g \, \mathbf{h}(t-1) +
        \mathbf{W}_{xh}^g \, \mathbf{x}(t) + \mathbf{b}^g  \right), \\
    \mathbf{h}(t) &= \mathbf{g}(t) \circ \mathbf{h}(t-1) + \left( 1 -
        \mathbf{g}(t) \right) \circ \mathbf{c}(t).
\end{aligned}
```

# Forward

    ugrnn(inp, state)
    ugrnn(inp)

## Arguments
- `inp`: The input to the ugrnn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the UGRNN. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct UGRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand UGRNN

function UGRNN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = UGRNNCell(input_size => hidden_size; kwargs...)
    return UGRNN{return_state, typeof(cell)}(cell)
end

function functor(rnn::UGRNN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> UGRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, ugrnn::UGRNN)
    print(io, "UGRNN(", size(ugrnn.cell.weight_ih, 2),
        " => ", size(ugrnn.cell.weight_ih, 1) ÷ 2)
    print(io, ")")
end
