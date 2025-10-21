#https://arxiv.org/abs/1711.06788
@doc raw"""
    MinimalRNNCell(input_size => hidden_size;
        init_encoder_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        init_memory_kernel = glorot_uniform,
        encoder_bias = true, recurrent_bias = true, memory_bias=true,
        independent_recurrence = false, integration_mode = :addition)

Minimal recurrent neural network unit [Chen2017](@cite).
See [`MinimalRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_encoder_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_memory_kernel`: initializer for the memory to hidden weights.
    Default is `glorot_uniform`.
- `encoder_bias`: include a bias in the encoder or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `memory_bias`: include memory to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \Phi(\mathbf{x}(t)) = \tanh\left( \mathbf{W}_{xz}
        \mathbf{x}(t) + \mathbf{b}^{z} \right), \\
    \mathbf{u}(t) &= \sigma\left( \mathbf{W}_{hh}^{u} \mathbf{h}(t-1) +
        \mathbf{W}_{zh}^{u} \mathbf{z}(t) + \mathbf{b}^{u} \right), \\
    \mathbf{h}(t) &= \mathbf{u}(t) \circ \mathbf{h}(t-1) + \left(1 -
        \mathbf{u}(t)\right) \circ \mathbf{z}(t)
\end{aligned}
```

# Forward

    minimalrnncell(inp, state)
    minimalrnncell(inp)

## Arguments
- `inp`: The input to the minimalrnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the MinimalRNNCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state.
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct MinimalRNNCell{I, H, Z, V, W, M, A} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    weight_mm::Z
    bias_ih::V
    bias_hh::W
    bias_mm::M
    integration_fn::A
end

@layer MinimalRNNCell

function MinimalRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_encoder_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        init_memory_kernel=glorot_uniform, encoder_bias::Bool=true,
        recurrent_bias::Bool=true, memory_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_encoder_kernel(hidden_size, input_size)
    if independent_recurrence
        weight_hh = vec(init_recurrent_kernel(hidden_size))
    else
        weight_hh = init_recurrent_kernel(hidden_size, hidden_size)
    end
    weight_mm = init_memory_kernel(hidden_size, hidden_size)
    bias_ih = create_bias(weight_ih, encoder_bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    bias_mm = create_bias(weight_mm, memory_bias, size(weight_mm, 1))
    integration_fn = _integration_fn(integration_mode)
    return MinimalRNNCell(weight_ih, weight_hh, weight_mm,
        bias_ih, bias_hh, bias_mm, integration_fn)
end

function (minimal::MinimalRNNCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(minimal, inp, 1 => size(minimal.weight_ih, 2))
    proj_ih = dense_proj(minimal.weight_ih, inp, minimal.bias_ih)
    proj_hh = dense_proj(minimal.weight_hh, state, minimal.bias_hh)
    proj_mm = dense_proj(minimal.weight_mm, c_state, minimal.bias_mm)
    new_cstate = tanh_fast.(proj_ih)
    update_gate = sigmoid_fast.(minimal.integration_fn(proj_hh, proj_mm))
    new_state = update_gate .* state .+
                (eltype(minimal.weight_ih)(1.0) .- update_gate) .* new_cstate
    return new_state, (new_state, new_cstate)
end

function initialstates(minimal::MinimalRNNCell)
    state = zeros_like(minimal.weight_hh, size(minimal.weight_hh, 1))
    second_state = zeros_like(minimal.weight_hh, size(minimal.weight_hh, 1))
    return state, second_state
end

function Base.show(io::IO, minimal::MinimalRNNCell)
    print(io, "MinimalRNNCell(", size(minimal.weight_ih, 2),
        " => ", size(minimal.weight_ih, 1), ")")
end

@doc raw"""
    MinimalRNN(input_size => hidden_size;
        return_state = false, kwargs...)

Minimal recurrent neural network [Chen2017](@cite).
See [`MinimalRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `return_state`: Option to return the last state together with the output.
  Default is `false`.
- `init_encoder_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_memory_kernel`: initializer for the memory to hidden weights.
    Default is `glorot_uniform`.
- `encoder_bias`: include a bias in the encoder or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `memory_bias`: include memory to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \Phi(\mathbf{x}(t)) = \tanh\left( \mathbf{W}_{xz}
        \mathbf{x}(t) + \mathbf{b}^{z} \right), \\
    \mathbf{u}(t) &= \sigma\left( \mathbf{W}_{hh}^{u} \mathbf{h}(t-1) +
        \mathbf{W}_{zh}^{u} \mathbf{z}(t) + \mathbf{b}^{u} \right), \\
    \mathbf{h}(t) &= \mathbf{u}(t) \circ \mathbf{h}(t-1) + \left(1 -
        \mathbf{u}(t)\right) \circ \mathbf{z}(t)
\end{aligned}
```

# Forward

    minimalrnn(inp, (state, c_state))
    minimalrnn(inp)

## Arguments
- `inp`: The input to the `minimalrnn`. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the `MinimalRNN`.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct MinimalRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand MinimalRNN

function MinimalRNN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = MinimalRNNCell(input_size => hidden_size; kwargs...)
    return MinimalRNN{return_state, typeof(cell)}(cell)
end

function functor(rnn::MinimalRNN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> MinimalRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, minimal::MinimalRNN)
    print(io, "MinimalRNN(", size(minimal.cell.weight_ih, 2),
        " => ", size(minimal.cell.weight_ih, 1))
    print(io, ")")
end
