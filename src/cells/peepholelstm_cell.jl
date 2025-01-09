#https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
@doc raw"""
    PeepholeLSTMCell((input_size => hidden_size)::Pair;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Peephole long short term memory cell](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf).
See [`PeepholeLSTM`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

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

    peepholelstmcell(inp, (state, cstate))
    peepholelstmcell(inp)

## Arguments

- `inp`: The input to the peepholelstmcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the PeepholeLSTMCell.
  They should be vectors of size `hidden_size` or matrices of size `hidden_size x batch_size`.
  If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state. 
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct PeepholeLSTMCell{I, H, V} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    bias::V
end
  
@layer PeepholeLSTMCell

function PeepholeLSTMCell(
    (input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true,
)
    Wi = init_kernel(hidden_size * 4, input_size)
    Wh = init_recurrent_kernel(hidden_size * 4, hidden_size)
    b = create_bias(Wi, bias, hidden_size * 4)
    return PeepholeLSTMCell(Wi, Wh, b)
end
  
function (lstm::PeepholeLSTMCell)(inp::AbstractVecOrMat, 
    (state, c_state))
    _size_check(lstm, inp, 1 => size(lstm.Wi, 2))
    b = lstm.bias
    g = lstm.Wi * inp .+ lstm.Wh * c_state .+ b
    input, forget, cell, output = chunk(g, 4; dims = 1)
    new_cstate = @. sigmoid_fast(forget) * c_state + sigmoid_fast(input) * tanh_fast(cell)
    new_state = @. sigmoid_fast(output) * tanh_fast(new_cstate)
    return new_cstate, (new_state, new_cstate)
end
  
Base.show(io::IO, lstm::PeepholeLSTMCell) =
    print(io, "PeepholeLSTMCell(", size(lstm.Wi, 2), " => ", size(lstm.Wi, 1) ÷ 4, ")")
  
  
@doc raw"""
    PeepholeLSTM((input_size => hidden_size)::Pair; kwargs...)

[Peephole long short term memory network](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf).
See [`PeepholeLSTMCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations

```math
\begin{align}
f_t &= \sigma_g(W_f x_t + U_f c_{t-1} + b_f), \\
i_t &= \sigma_g(W_i x_t + U_i c_{t-1} + b_i), \\
o_t &= \sigma_g(W_o x_t + U_o c_{t-1} + b_o), \\
c_t &= f_t \odot c_{t-1} + i_t \odot \sigma_c(W_c x_t + b_c), \\
h_t &= o_t \odot \sigma_h(c_t).
\end{align}
```
# Forward

    peepholelstm(inp, (state, cstate))
    peepholelstm(inp)

## Arguments
- `inp`: The input to the peepholelstm. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the PeepholeLSTM. 
  They should be vectors of size `hidden_size` or matrices of size `hidden_size x batch_size`.
  If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
"""
struct PeepholeLSTM{M} <: AbstractRecurrentLayer
    cell::M
end

@layer :noexpand PeepholeLSTM

function PeepholeLSTM((input_size, hidden_size)::Pair; kwargs...)
    cell = PeepholeLSTMCell(input_size => hidden_size; kwargs...)
    return PeepholeLSTM(cell)
end

function Base.show(io::IO, peepholelstm::PeepholeLSTM)
    print(io, "PeepholeLSTM(", size(peepholelstm.cell.Wi, 2), " => ", size(peepholelstm.cell.Wi, 1))
    print(io, ")")
end