#https://arxiv.org/abs/2109.00020
@doc raw"""
    WMCLSTMCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        init_memory_kernel = glorot_uniform,
        bias = true, recurrent_bias = true, memory_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Long short term memory cell with working memory
connections [Landi2021](@cite).
See [`WMCLSTM`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_memory_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `memory_bias`: include memory to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.

# Equations

```math
\begin{aligned}
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{hh} \mathbf{h}(t-1) +
        \tanh\left( \mathbf{W}^{i}_{ch} \mathbf{c}(t-1) \right) +
        \mathbf{b}^{i} \right) \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) +
        \tanh\left( \mathbf{W^{f}_{ch}} \mathbf{c}(t-1) \right) +
        \mathbf{b}^{f} \right) \\
    \mathbf{o}(t) &= \sigma\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{hh} \mathbf{h}(t-1) +
        \tanh\left( \mathbf{W}^{o}_{ch} \mathbf{c}(t) \right) +
        \mathbf{b}^{o} \right) \\
    \mathbf{c}(t) &= \mathbf{f}(t) \circ \mathbf{c}(t-1) + \mathbf{i}(t) \circ
        \sigma_{c}\left( \mathbf{W}^{c}_{ih} \mathbf{x}(t) + \mathbf{b}^{c} \right) \\
    \mathbf{h}(t) &= \mathbf{o}(t) \circ \sigma_{h}\left( \mathbf{c}(t) \right)
\end{aligned}
```

# Forward

    wmclstmcell(inp, (state, cstate))
    wmclstmcell(inp)

## Arguments

- `inp`: The input to the wmclstmcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the WMCLSTMCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state.
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct WMCLSTMCell{I, H, M, V, W, U, A} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    weight_mh::M
    bias_ih::V
    bias_hh::W
    bias_mh::U
    integration_fn::A
end

@layer WMCLSTMCell

function WMCLSTMCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        init_memory_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true, memory_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(4 * hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 4)
    weight_mh = init_memory_kernel(3 * hidden_size, hidden_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    bias_mh = create_bias(weight_mh, memory_bias, size(weight_mh, 1))
    integration_fn = _integration_fn(integration_mode)
    return WMCLSTMCell(weight_ih, weight_hh, weight_mh, bias_ih, bias_hh,
        bias_mh, integration_fn)
end

function (lstm::WMCLSTMCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(lstm, inp, 1 => size(lstm.weight_ih, 2))
    proj_ih = dense_proj(lstm.weight_ih, inp, lstm.bias_ih)
    proj_hh = dense_proj(lstm.weight_hh, state, lstm.bias_hh)
    fused_gates = lstm.integration_fn(proj_ih, proj_hh)
    wms = chunk(lstm.weight_mh, 3; dims=1) #memory_matrices
    bms = chunk(lstm.bias_mh, 3; dims=1)
    proj_mh_1 = dense_proj(wms[1], c_state, bms[1]) #memorygates
    proj_mh_2 = dense_proj(wms[2], c_state, bms[2]) #memorygates
    gates = chunk(fused_gates, 4; dims=1)
    input_gate = @. sigmoid_fast(gates[1] + tanh_fast(proj_mh_1))
    forget_gate = @. sigmoid_fast(gates[2] + tanh_fast(proj_mh_2))
    cell_gate = @. tanh_fast(gates[4])
    new_cstate = @. forget_gate * c_state + input_gate * cell_gate
    proj_mh_2 = dense_proj(wms[3], new_cstate, bms[3])
    output_gate = @. sigmoid_fast(gates[3] + tanh_fast(proj_mh_2))
    new_state = @. output_gate * tanh_fast(new_cstate)
    return new_state, (new_state, new_cstate)
end

function initialstates(lstm::WMCLSTMCell)
    state = zeros_like(lstm.weight_hh, size(lstm.weight_hh, 1) ÷ 4)
    second_state = zeros_like(lstm.weight_hh, size(lstm.weight_hh, 1) ÷ 4)
    return state, second_state
end

function Base.show(io::IO, lstm::WMCLSTMCell)
    print(io, "WMCLSTMCell(", size(lstm.weight_ih, 2),
        " => ", size(lstm.weight_ih, 1) ÷ 4, ")")
end

@doc raw"""
    WMCLSTM(input_size => hidden_size;
        return_state=false,
        kwargs...)

Long short term memory cell with working memory
connections [Landi2021](@cite).
See [`WMCLSTM`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_memory_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `memory_bias`: include memory to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `return_state`: Option to return the last state together with the output.
  Default is `false`.

# Equations

```math
\begin{aligned}
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{hh} \mathbf{h}(t-1) +
        \tanh\left( \mathbf{W}^{i}_{ch} \mathbf{c}(t-1) \right) +
        \mathbf{b}^{i} \right) \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) +
        \tanh\left( \mathbf{W^{f}_{ch}} \mathbf{c}(t-1) \right) +
        \mathbf{b}^{f} \right) \\
    \mathbf{o}(t) &= \sigma\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{hh} \mathbf{h}(t-1) +
        \tanh\left( \mathbf{W}^{o}_{ch} \mathbf{c}(t) \right) +
        \mathbf{b}^{o} \right) \\
    \mathbf{c}(t) &= \mathbf{f}(t) \circ \mathbf{c}(t-1) + \mathbf{i}(t) \circ
        \sigma_{c}\left( \mathbf{W}^{c}_{ih} \mathbf{x}(t) + \mathbf{b}^{c} \right) \\
    \mathbf{h}(t) &= \mathbf{o}(t) \circ \sigma_{h}\left( \mathbf{c}(t) \right)
\end{aligned}
```

# Forward

    wmclstm(inp, (state, cstate))
    wmclstm(inp)

## Arguments
- `inp`: The input to the wmclstm. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the WMCLSTM.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct WMCLSTM{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand WMCLSTM

function WMCLSTM((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = WMCLSTMCell(input_size => hidden_size; kwargs...)
    return WMCLSTM{return_state, typeof(cell)}(cell)
end

function functor(rnn::WMCLSTM{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> WMCLSTM{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, lstm::WMCLSTM)
    print(io, "WMCLSTM(", size(lstm.cell.weight_ih, 2),
        " => ", size(lstm.cell.weight_ih, 1) ÷ 4)
    print(io, ")")
end
