#https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
@doc raw"""
    PeepholeLSTMCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        init_peephole_kernel = glorot_uniform,
        bias = true)

[Peephole long short term memory cell](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf).
See [`PeepholeLSTM`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_peephole_kernel`: initializer for the hidden to peephole weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations

```math
\begin{aligned}
    z_t &= \tanh(W_z x_t + U_z h_{t-1} + b_z), \\
    i_t &= \sigma(W_i x_t + U_i h_{t-1} + p_i \odot c_{t-1} + b_i), \\
    f_t &= \sigma(W_f x_t + U_f h_{t-1} + p_f \odot c_{t-1} + b_f), \\
    c_t &= f_t \odot c_{t-1} + i_t \odot z_t, \\
    o_t &= \sigma(W_o x_t + U_o h_{t-1} + p_o \odot c_t + b_o), \\
    h_t &= o_t \odot \tanh(c_t).
\end{aligned}
```

# Forward

    peepholelstmcell(inp, (state, cstate))
    peepholelstmcell(inp)

## Arguments

- `inp`: The input to the peepholelstmcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the PeepholeLSTMCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state. 
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct PeepholeLSTMCell{I, H, P, V} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    Wp::P
    bias::V
end

@layer PeepholeLSTMCell

function PeepholeLSTMCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        init_peephole_kernel=glorot_uniform, bias::Bool=true)
    Wi = init_kernel(hidden_size * 4, input_size)
    Wh = init_recurrent_kernel(hidden_size * 4, hidden_size)
    Wp = init_peephole_kernel(hidden_size * 3)
    b = create_bias(Wi, bias, hidden_size * 4)
    return PeepholeLSTMCell(Wi, Wh, vec(Wp), b)
end

function (lstm::PeepholeLSTMCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(lstm, inp, 1 => size(lstm.Wi, 2))
    b = lstm.bias
    gates = lstm.Wi * inp .+ lstm.Wh * state .+ b
    input, forget, cell, output = chunk(gates, 4; dims=1)
    gpeep = chunk(lstm.Wp, 3; dims=1)
    new_cstate = @. sigmoid_fast(forget + gpeep[1] * c_state) * c_state +
                    sigmoid_fast(input + gpeep[2] * c_state) * tanh_fast(cell)
    new_state = @. sigmoid_fast(output + gpeep[3] * c_state) * tanh_fast(new_cstate)
    return new_state, (new_state, new_cstate)
end

function Base.show(io::IO, lstm::PeepholeLSTMCell)
    print(io, "PeepholeLSTMCell(", size(lstm.Wi, 2), " => ", size(lstm.Wi, 1) ÷ 4, ")")
end

@doc raw"""
    PeepholeLSTM(input_size => hidden_size;
        return_state=false,
        kwargs...)

[Peephole long short term memory network](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf).
See [`PeepholeLSTMCell`](@ref) for a layer that processes a single sequence.

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
f_t &= \sigma_g(W_f x_t + U_f c_{t-1} + b_f), \\
i_t &= \sigma_g(W_i x_t + U_i c_{t-1} + b_i), \\
o_t &= \sigma_g(W_o x_t + U_o c_{t-1} + b_o), \\
c_t &= f_t \odot c_{t-1} + i_t \odot \sigma_c(W_c x_t + b_c), \\
h_t &= o_t \odot \sigma_h(c_t).
\end{aligned}
```

# Forward

    peepholelstm(inp, (state, cstate))
    peepholelstm(inp)

## Arguments
- `inp`: The input to the peepholelstm. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the PeepholeLSTM. 
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct PeepholeLSTM{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand PeepholeLSTM

function PeepholeLSTM((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = PeepholeLSTMCell(input_size => hidden_size; kwargs...)
    return PeepholeLSTM{return_state, typeof(cell)}(cell)
end

function functor(rnn::PeepholeLSTM{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> PeepholeLSTM{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, peepholelstm::PeepholeLSTM)
    print(io, "PeepholeLSTM(", size(peepholelstm.cell.Wi, 2),
        " => ", size(peepholelstm.cell.Wi, 1) ÷ 4)
    print(io, ")")
end
