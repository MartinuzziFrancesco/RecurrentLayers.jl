#https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
@doc raw"""
    OriginalLSTMCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

Original long short term memory cell [Hochreiter1997](@cite) with no forget gate.
See [`OriginalLSTM`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \tanh\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{z} \right), \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{i} \right), \\
    \mathbf{c}(t) &= \mathbf{c}(t-1) + \mathbf{i}(t)
        \odot \mathbf{z}(t), \\
    \mathbf{o}(t) &= \sigma\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{o} \right), \\
    \mathbf{h}(t) &= \mathbf{o}(t) \odot \tanh\left( \mathbf{c}(t) \right)
\end{aligned}
```

# Forward

    originallstmcell(inp, (state, cstate))
    originallstmcell(inp)

## Arguments

- `inp`: The input to the originallstmcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the OriginalLSTMCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state.
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct OriginalLSTMCell{I, H, V} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

@layer OriginalLSTMCell

function OriginalLSTMCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size * 3, input_size)
    Wh = init_recurrent_kernel(hidden_size * 3, hidden_size)
    b = create_bias(Wi, bias, hidden_size * 3)
    return OriginalLSTMCell(Wi, Wh, b)
end

function (lstm::OriginalLSTMCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(lstm, inp, 1 => size(lstm.Wi, 2))
    b = lstm.bias
    gates = lstm.Wi * inp .+ lstm.Wh * state .+ b
    input, cell, output = chunk(gates, 3; dims=1)
    new_cstate = @. c_state + sigmoid_fast(input) * tanh_fast(cell)
    new_state = @. sigmoid_fast(output) * tanh_fast(new_cstate)
    return new_state, (new_state, new_cstate)
end

function Base.show(io::IO, lstm::OriginalLSTMCell)
    print(io, "OriginalLSTMCell(", size(lstm.Wi, 2), " => ", size(lstm.Wi, 1) ÷ 3, ")")
end

@doc raw"""
    OriginalLSTM(input_size => hidden_size;
        return_state=false,
        kwargs...)

Original long short term memory network [Hochreiter1997](@cite).
See [`OriginalLSTMCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.
- `return_state`: Option to return the last state together with the output.
  Default is `false`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \tanh\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{z} \right), \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{i} \right), \\
    \mathbf{c}(t) &= \mathbf{c}(t-1) + \mathbf{i}(t)
        \odot \mathbf{z}(t), \\
    \mathbf{o}(t) &= \sigma\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{o} \right), \\
    \mathbf{h}(t) &= \mathbf{o}(t) \odot \tanh\left( \mathbf{c}(t) \right)
\end{aligned}
```

# Forward

    originallstm(inp, (state, cstate))
    originallstm(inp)

## Arguments
- `inp`: The input to the originallstm. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the OriginalLSTM.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct OriginalLSTM{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand OriginalLSTM

function OriginalLSTM((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = OriginalLSTMCell(input_size => hidden_size; kwargs...)
    return OriginalLSTM{return_state, typeof(cell)}(cell)
end

function functor(rnn::OriginalLSTM{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> OriginalLSTM{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, originallstm::OriginalLSTM)
    print(io, "OriginalLSTM(", size(originallstm.cell.Wi, 2),
        " => ", size(originallstm.cell.Wi, 1) ÷ 4)
    print(io, ")")
end
