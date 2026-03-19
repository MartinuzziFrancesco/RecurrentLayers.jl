@doc raw"""
    MCLSTMCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Memory controller long short term memory cell [Ben2017](@cite).
See [`MCLSTM`](@ref) for a layer that processes entire sequences.

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
\begin{align}
    \mathbf{f}_t &= \sigma\left( \mathbf{W}_{fv} \mathbf{v}_{t-1} +
        \mathbf{W}_{fx} \mathbf{x}_t + \mathbf{b}_f \right), \\
    \mathbf{i}_t &= \sigma\left( \mathbf{W}_{iv} \mathbf{v}_{t-1} +
        \mathbf{W}_{ix} \mathbf{x}_t + \mathbf{b}_i \right), \\
    \mathbf{o}_t &= \sigma\left( \mathbf{W}_{ov} \mathbf{v}_{t-1} +
        \mathbf{W}_{ox} \mathbf{x}_t + \mathbf{b}_o \right), \\
    \mathbf{m}_t &= \sigma\left( \mathbf{W}_{mv} \mathbf{v}_{t-1} +
        \mathbf{W}_{mx} \mathbf{x}_t + \mathbf{b}_m \right), \\
    \mathbf{n}_t &= \tanh\left( \mathbf{W}_{nv} \mathbf{v}_{t-1} +
        \mathbf{W}_{nx} \mathbf{x}_t + \mathbf{b}_n \right), \\
    \mathbf{c}_t &= \mathbf{f}_t \circ \mathbf{c}_{t-1} + \mathbf{i}_t
        \circ \mathbf{n}_t, \\
    \mathbf{h}_t &= \mathbf{o}_t \circ \tanh(\mathbf{c}_t), \\
    \mathbf{v}_t &= \mathbf{m}_t \circ \tanh(\mathbf{c}_t).
\end{align}
```

# Forward

    mclstmcell(inp, (state, cstate))
    mclstmcell(inp)

## Arguments

- `inp`: The input to the mclstmcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the MCLSTMCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state.
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct MCLSTMCell{I, H, V, W, A} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer MCLSTMCell

function MCLSTMCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(5 * hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 5)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)
    return MCLSTMCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (lstm::MCLSTMCell)(inp::AbstractVecOrMat, (v_state, c_state))
    _size_check(lstm, inp, 1 => size(lstm.weight_ih, 2))
    proj_ih = dense_proj(lstm.weight_ih, inp, lstm.bias_ih)
    proj_hh = dense_proj(lstm.weight_hh, v_state, lstm.bias_hh)
    gates = lstm.integration_fn(proj_ih, proj_hh)
    fg, ig, og, mg, cell = chunk(gates, 5; dims=1)
    new_cstate = @. fg * c_state + ig * cell
    new_state = og .* tanh_fast.(new_cstate)
    new_vstate = mg .* tanh_fast.(new_cstate)
    return new_state, (new_vstate, new_cstate)
end

function initialstates(lstm::MCLSTMCell)
    state = zeros_like(lstm.weight_hh, size(lstm.weight_hh, 1) ÷ 5)
    second_state = zeros_like(lstm.weight_hh, size(lstm.weight_hh, 1) ÷ 5)
    return state, second_state
end

function Base.show(io::IO, lstm::MCLSTMCell)
    print(io, "MCLSTMCell(", size(lstm.weight_ih, 2),
        " => ", size(lstm.weight_ih, 1) ÷ 5, ")")
end

@doc raw"""
    MCLSTM(iinput_size => hidden_size;
        return_state=false,
        kwargs...)

Memory controller long short term memory cell [Ben2017](@cite).
See [`MCLSTMCell`](@ref) for a layer that processes a single sequence.

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
\begin{align}
    \mathbf{f}_t &= \sigma\left( \mathbf{W}_{fv} \mathbf{v}_{t-1} +
        \mathbf{W}_{fx} \mathbf{x}_t + \mathbf{b}_f \right), \\
    \mathbf{i}_t &= \sigma\left( \mathbf{W}_{iv} \mathbf{v}_{t-1} +
        \mathbf{W}_{ix} \mathbf{x}_t + \mathbf{b}_i \right), \\
    \mathbf{o}_t &= \sigma\left( \mathbf{W}_{ov} \mathbf{v}_{t-1} +
        \mathbf{W}_{ox} \mathbf{x}_t + \mathbf{b}_o \right), \\
    \mathbf{m}_t &= \sigma\left( \mathbf{W}_{mv} \mathbf{v}_{t-1} +
        \mathbf{W}_{mx} \mathbf{x}_t + \mathbf{b}_m \right), \\
    \mathbf{n}_t &= \tanh\left( \mathbf{W}_{nv} \mathbf{v}_{t-1} +
        \mathbf{W}_{nx} \mathbf{x}_t + \mathbf{b}_n \right), \\
    \mathbf{c}_t &= \mathbf{f}_t \circ \mathbf{c}_{t-1} + \mathbf{i}_t
        \circ \mathbf{n}_t, \\
    \mathbf{h}_t &= \mathbf{o}_t \circ \tanh(\mathbf{c}_t), \\
    \mathbf{v}_t &= \mathbf{m}_t \circ \tanh(\mathbf{c}_t).
\end{align}
```

# Forward

    mclstm(inp, (state, cstate))
    mclstm(inp)

## Arguments
- `inp`: The input to the mclstm. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the MCLSTM.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct MCLSTM{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand MCLSTM

function MCLSTM((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = MCLSTMCell(input_size => hidden_size; kwargs...)
    return MCLSTM{return_state, typeof(cell)}(cell)
end

function functor(rnn::MCLSTM{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> MCLSTM{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, lstm::MCLSTM)
    print(io, "MCLSTM(", size(lstm.cell.weight_ih, 2),
        " => ", size(lstm.cell.weight_ih, 1) ÷ 5)
    print(io, ")")
end
