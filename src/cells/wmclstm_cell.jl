#https://arxiv.org/abs/2109.00020
@doc raw"""
    WMCLSTMCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        init_memory_kernel = glorot_uniform,
        bias = true)

[Long short term memory cell with working memory
connections](https://arxiv.org/abs/2109.00020).
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
- `bias`: include a bias or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{i}_t &= \sigma\left(\mathbf{W}_{ix} \mathbf{x}_t + \mathbf{W}_{ih}
        \mathbf{h}_{t-1} + \tanh(\mathbf{W}_{ic} \mathbf{c}_{t-1}) +
        \mathbf{b}_i\right), \\
    \mathbf{f}_t &= \sigma\left(\mathbf{W}_{fx} \mathbf{x}_t + \mathbf{W}_{fh}
        \mathbf{h}_{t-1} + \tanh(\mathbf{W}_{fc} \mathbf{c}_{t-1}) +
        \mathbf{b}_f\right), \\
    \mathbf{o}_t &= \sigma\left(\mathbf{W}_{ox} \mathbf{x}_t + \mathbf{W}_{oh}
        \mathbf{h}_{t-1} + \tanh(\mathbf{W}_{oc} \mathbf{c}_t) + \mathbf{b}_o\right), \\
    \mathbf{c}_t &= \mathbf{f}_t \circ \mathbf{c}_{t-1} + \mathbf{i}_t \circ
        \sigma_c(\mathbf{W}_{c} \mathbf{x}_t + \mathbf{b}_c), \\
    \mathbf{h}_t &= \mathbf{o}_t \circ \sigma_h(\mathbf{c}_t).
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
struct WMCLSTMCell{I, H, M, V} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    Wm::M
    bias::V
end

@layer WMCLSTMCell

function WMCLSTMCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        init_memory_kernel=glorot_uniform, bias::Bool=true)
    Wi = init_kernel(hidden_size * 4, input_size)
    Wh = init_recurrent_kernel(hidden_size * 4, hidden_size)
    Wm = init_memory_kernel(hidden_size * 3, hidden_size)
    b = create_bias(Wi, bias, hidden_size * 4)
    return WMCLSTMCell(Wi, Wh, Wm, b)
end

function (lstm::WMCLSTMCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(lstm, inp, 1 => size(lstm.Wi, 2))
    fused_gates = lstm.Wi * inp .+ lstm.Wh * c_state .+ lstm.bias
    memory_matrices = chunk(lstm.Wm, 3; dims=1)
    memory_gates = memory_matrices[1] * c_state, memory_matrices[2] * c_state
    gates = chunk(fused_gates, 4; dims=1)
    input_gate = @. sigmoid_fast(gates[1] + tanh_fast(memory_gates[1]))
    forget_gate = @. sigmoid_fast(gates[2] + tanh_fast(memory_gates[2]))
    cell_gate = @. tanh_fast(gates[4])
    new_cstate = @. forget_gate * c_state + input_gate * cell_gate
    memory_gate = memory_matrices[3] * new_cstate
    output_gate = @. sigmoid_fast(gates[3] + tanh_fast(memory_gate))
    new_state = @. output_gate * tanh_fast(new_cstate)
    return new_state, (new_state, new_cstate)
end

function Base.show(io::IO, lstm::WMCLSTMCell)
    print(io, "WMCLSTMCell(", size(lstm.Wi, 2), " => ", size(lstm.Wi, 1) ÷ 4, ")")
end

@doc raw"""
    WMCLSTM(input_size => hidden_size;
        return_state=false,
        kwargs...)

[Long short term memory cell with working memory
connections](https://arxiv.org/abs/2109.00020).
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
- `bias`: include a bias or not. Default is `true`.
- `return_state`: Option to return the last state together with the output.
  Default is `false`.

# Equations

```math
\begin{aligned}
    \mathbf{i}_t &= \sigma\left(\mathbf{W}_{ix} \mathbf{x}_t + \mathbf{W}_{ih}
        \mathbf{h}_{t-1} + \tanh(\mathbf{W}_{ic} \mathbf{c}_{t-1}) +
        \mathbf{b}_i\right), \\
    \mathbf{f}_t &= \sigma\left(\mathbf{W}_{fx} \mathbf{x}_t + \mathbf{W}_{fh}
        \mathbf{h}_{t-1} + \tanh(\mathbf{W}_{fc} \mathbf{c}_{t-1}) +
        \mathbf{b}_f\right), \\
    \mathbf{o}_t &= \sigma\left(\mathbf{W}_{ox} \mathbf{x}_t + \mathbf{W}_{oh}
        \mathbf{h}_{t-1} + \tanh(\mathbf{W}_{oc} \mathbf{c}_t) + \mathbf{b}_o\right), \\
    \mathbf{c}_t &= \mathbf{f}_t \circ \mathbf{c}_{t-1} + \mathbf{i}_t \circ
        \sigma_c(\mathbf{W}_{c} \mathbf{x}_t + \mathbf{b}_c), \\
    \mathbf{h}_t &= \mathbf{o}_t \circ \sigma_h(\mathbf{c}_t).
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
    print(io, "WMCLSTM(", size(lstm.cell.Wi, 2),
        " => ", size(lstm.cell.Wi, 1) ÷ 4)
    print(io, ")")
end
