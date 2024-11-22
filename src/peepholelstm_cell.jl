#https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
struct PeepholeLSTMCell{I, H, V}
    Wi::I
    Wh::H
    bias::V
end
  
Flux.@layer PeepholeLSTMCell

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
\begin{align}
f_t &= \sigma_g(W_f x_t + U_f c_{t-1} + b_f), \\
i_t &= \sigma_g(W_i x_t + U_i c_{t-1} + b_i), \\
o_t &= \sigma_g(W_o x_t + U_o c_{t-1} + b_o), \\
c_t &= f_t \odot c_{t-1} + i_t \odot \sigma_c(W_c x_t + b_c), \\
h_t &= o_t \odot \sigma_h(c_t).
\end{align}
```

# Forward

    lstmcell(x, [h, c])

The forward pass takes the following arguments:

- `x`: Input to the cell, which can be a vector of size `in` or a matrix of size `in x batch_size`.
- `h`: The hidden state vector of the cell, sized `out`, or a matrix of size `out x batch_size`.
- `c`: The candidate state, sized `out`, or a matrix of size `out x batch_size`.
If not provided, both `h` and `c` default to vectors of zeros.

# Examples

"""
function PeepholeLSTMCell(
    (input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true,
)
    Wi = init_kernel(hidden_size * 4, input_size)
    Wh = init_recurrent_kernel(hidden_size * 4, hidden_size)
    b = create_bias(Wi, bias, hidden_size * 4)
    cell = PeepholeLSTMCell(Wi, Wh, b)
    return cell
end
  
function (lstm::PeepholeLSTMCell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(lstm.Wh, 2))
    c_state = zeros_like(state)
    return lstm(inp, (state, c_state))
end
  
function (lstm::PeepholeLSTMCell)(inp::AbstractVecOrMat, 
    (state, c_state))
    _size_check(lstm, inp, 1 => size(lstm.Wi, 2))
    b = lstm.bias
    g = lstm.Wi * inp .+ lstm.Wh * c_state .+ b
    input, forget, cell, output = chunk(g, 4; dims = 1)
    new_cstate = @. sigmoid_fast(forget) * c_state + sigmoid_fast(input) * tanh_fast(cell)
    new_state = @. sigmoid_fast(output) * tanh_fast(new_cstate)
    return new_state, new_cstate
end
  
Base.show(io::IO, lstm::PeepholeLSTMCell) =
    print(io, "PeepholeLSTMCell(", size(lstm.Wi, 2), " => ", size(lstm.Wi, 1) ÷ 4, ")")
  
  

struct PeepholeLSTM{M}
    cell::M
end

Flux.@layer :expand PeepholeLSTM

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
"""
function PeepholeLSTM((input_size, hidden_size)::Pair; kwargs...)
    cell = PeepholeLSTM(input_size => hidden_size; kwargs...)
    return PeepholeLSTM(cell)
end

function (lstm::PeepholeLSTM)(inp)
    state = zeros_like(inp, size(lstm.cell.Wh, 2))
    c_state = zeros_like(state)
    return lstm(inp, (state, c_state))
end

function (lstm::PeepholeLSTM)(inp, (state, c_state))
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    new_cstate = []
    for inp_t in eachslice(inp, dims=2)
        state, c_state = nas.cell(inp_t, (state, c_state))
        new_state = vcat(new_state, [state])
        new_cstate = vcat(new_cstate, [c_state])
    end
    return stack(new_state, dims=2), stack(new_cstate, dims=2)
end
