#https://arxiv.org/abs/1609.07959
@doc raw"""
    MultiplicativeLSTMCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        init_multiplicative_kernel=glorot_uniform,
        bias = true, recurrent_bias = true, multiplicative_bias=true,
        independent_recurrence = false, integration_mode = :addition)

Multiplicative long short term memory cell [Krause2017](@cite).
See [`MultiplicativeLSTM`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_multiplicative_kernel`: initializer for the multiplicative to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `multiplicative_bias`: include multiplicative bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.


# Equations

```math
\begin{aligned}
    \mathbf{m}(t) &= \left( \mathbf{W}^{m}_{ih} \mathbf{x}(t) \right) \circ
        \left( \mathbf{W}^{m}_{hh} \mathbf{h}(t-1) \right), \\
    \hat{\mathbf{h}}(t) &= \mathbf{W}^{h}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{h}_{mh} \mathbf{m}(t) + \mathbf{b}^{h}, \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{mh} \mathbf{m}(t) + \mathbf{b}^{i} \right), \\
    \mathbf{o}(t) &= \sigma\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{mh} \mathbf{m}(t) + \mathbf{b}^{o} \right), \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{mh} \mathbf{m}(t) + \mathbf{b}^{f} \right), \\
    \mathbf{c}(t) &= \mathbf{f}(t) \circ \mathbf{c}(t-1) + \mathbf{i}(t)
        \circ \tanh\left( \hat{\mathbf{h}}(t) \right), \\
    \mathbf{h}(t) &= \tanh\left( \mathbf{c}(t) \right) \circ \mathbf{o}(t)
\end{aligned}
```

# Forward

    multiplicativelstmcell(inp, (state, cstate))
    multiplicativelstmcell(inp)

## Arguments

- `inp`: The input to the multiplicativelstmcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the MultiplicativeLSTMCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state.
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct MultiplicativeLSTMCell{I, H, M, V, W, N, A} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    weight_mh::M
    bias_ih::V
    bias_hh::W
    bias_mh::N
    integration_fn::A
end

@layer MultiplicativeLSTMCell

function MultiplicativeLSTMCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        init_multiplicative_kernel=glorot_uniform, bias::Bool=true,
        multiplicative_bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(hidden_size * 5, input_size)
    if independent_recurrence
        weight_hh = vec(init_recurrent_kernel(hidden_size))
    else
        weight_hh = init_recurrent_kernel(hidden_size, hidden_size)
    end
    weight_mh = init_multiplicative_kernel(hidden_size * 4, hidden_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    bias_mh = create_bias(weight_mh, multiplicative_bias, size(weight_mh, 1))
    integration_fn = _integration_fn(integration_mode)
    return MultiplicativeLSTMCell(weight_ih, weight_hh, weight_mh, bias_ih, bias_hh,
        bias_mh, integration_fn)
end

function (lstm::MultiplicativeLSTMCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(lstm, inp, 1 => size(lstm.weight_ih, 2))
    proj_ih = dense_proj(lstm.weight_ih, inp, lstm.bias_ih)
    proj_hh = dense_proj(lstm.weight_hh, state, lstm.bias_hh)
    gxs = chunk(proj_ih, 5; dims=1)
    multiplicative_state = gxs[1] .* proj_hh
    proj_mh = dense_proj(lstm.weight_mh, multiplicative_state, lstm.bias_mh)
    gms = chunk(proj_mh, 4; dims=1)
    input_gate = sigmoid_fast.(lstm.integration_fn(gxs[2], gms[1]))
    output_gate = sigmoid_fast.(lstm.integration_fn(gxs[3], gms[2]))
    forget_gate = sigmoid_fast.(lstm.integration_fn(gxs[4], gms[3]))
    candidate_state = tanh_fast.(lstm.integration_fn(gxs[5], gms[4]))
    new_cstate = @. forget_gate * c_state + input_gate * candidate_state
    new_state = @. output_gate * tanh_fast(new_cstate)
    return new_state, (new_state, new_cstate)
end

function initialstates(lstm::MultiplicativeLSTMCell)
    state = zeros_like(lstm.weight_hh, size(lstm.weight_hh, 1))
    second_state = zeros_like(lstm.weight_hh, size(lstm.weight_hh, 1))
    return state, second_state
end

function Base.show(io::IO, lstm::MultiplicativeLSTMCell)
    print(
        io, "MultiplicativeLSTMCell(", size(lstm.weight_ih, 2),
        " => ", size(lstm.weight_ih, 1) ÷ 5, ")")
end

@doc raw"""
    MultiplicativeLSTM(input_size => hidden_size;
        return_state=false,
        kwargs...)

Multiplicative long short term memory network [Krause2017](@cite).
See [`MultiplicativeLSTMCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_multiplicative_kernel`: initializer for the multiplicative to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `multiplicative_bias`: include multiplicative bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `return_state`: Option to return the last state together with the output.
  Default is `false`.

# Equations

```math
\begin{aligned}
    \mathbf{m}(t) &= \left( \mathbf{W}^{m}_{ih} \mathbf{x}(t) \right) \circ
        \left( \mathbf{W}^{m}_{hh} \mathbf{h}(t-1) \right), \\
    \hat{\mathbf{h}}(t) &= \mathbf{W}^{h}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{h}_{mh} \mathbf{m}(t) + \mathbf{b}^{h}, \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{mh} \mathbf{m}(t) + \mathbf{b}^{i} \right), \\
    \mathbf{o}(t) &= \sigma\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{mh} \mathbf{m}(t) + \mathbf{b}^{o} \right), \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{mh} \mathbf{m}(t) + \mathbf{b}^{f} \right), \\
    \mathbf{c}(t) &= \mathbf{f}(t) \circ \mathbf{c}(t-1) + \mathbf{i}(t)
        \circ \tanh\left( \hat{\mathbf{h}}(t) \right), \\
    \mathbf{h}(t) &= \tanh\left( \mathbf{c}(t) \right) \circ \mathbf{o}(t)
\end{aligned}
```

# Forward

    multiplicativelstm(inp, (state, cstate))
    multiplicativelstm(inp)

## Arguments
- `inp`: The input to the multiplicativelstm. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the MultiplicativeLSTM.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct MultiplicativeLSTM{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand MultiplicativeLSTM

function MultiplicativeLSTM((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = MultiplicativeLSTMCell(input_size => hidden_size; kwargs...)
    return MultiplicativeLSTM{return_state, typeof(cell)}(cell)
end

function functor(rnn::MultiplicativeLSTM{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> MultiplicativeLSTM{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, lstm::MultiplicativeLSTM)
    print(io, "MultiplicativeLSTM(", size(lstm.cell.weight_ih, 2),
        " => ", size(lstm.cell.weight_ih, 1) ÷ 5)
    print(io, ")")
end
