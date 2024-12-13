#https://arxiv.org/pdf/1705.07393
struct RANCell{I,H,V} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer RANCell

@doc raw"""
    RANCell((input_size => hidden_size)::Pair;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

The `RANCell`, introduced in [this paper](https://arxiv.org/pdf/1705.07393), 
is a recurrent cell layer which provides additional memory through the
use of gates.

and returns both h_t anf c_t.

See [`RAN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
\tilde{c}_t &= W_c x_t, \\
i_t         &= \sigma(W_i x_t + U_i h_{t-1} + b_i), \\
f_t         &= \sigma(W_f x_t + U_f h_{t-1} + b_f), \\
c_t         &= i_t \odot \tilde{c}_t + f_t \odot c_{t-1}, \\
h_t         &= g(c_t)
\end{aligned}
```

# Forward

    rancell(x, [h, c])

The forward pass takes the following arguments:

- `x`: Input to the cell, which can be a vector of size `in` or a matrix of size `in x batch_size`.
- `h`: The hidden state vector of the cell, sized `out`, or a matrix of size `out x batch_size`.
- `c`: The candidate state, sized `out`, or a matrix of size `out x batch_size`.
If not provided, both `h` and `c` default to vectors of zeros.

# Examples

```julia
rancell = RANCell(3 => 5)
inp = rand(Float32, 3)
#initializing the hidden states, if we want to provide them
state = rand(Float32, 5)
c_state = rand(Float32, 5)

#result with default initialization of internal states
result = rancell(inp)
#result with internal states provided
result_state = rancell(inp, (state, c_state))
```
"""
function RANCell((input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true)
    Wi = init_kernel(3 * hidden_size, input_size)
    Wh = init_recurrent_kernel(2 * hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wh, 1))
    return RANCell(Wi, Wh, b)
end

function (ran::RANCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(ran, inp, 1 => size(ran.Wi,2))
    Wi, Wh, b = ran.Wi, ran.Wh, ran.bias

    #split
    gxs = chunk(Wi * inp, 3; dims=1)
    ghs = chunk(Wh * state .+ b, 2; dims=1)

    #compute
    input_gate = @. sigmoid_fast(gxs[2] + ghs[1])
    forget_gate = @. sigmoid_fast(gxs[3] + ghs[2])
    candidate_state = @. input_gate * gxs[1] + forget_gate * c_state
    new_state = tanh_fast(candidate_state)
    return new_state, candidate_state
end

Base.show(io::IO, ran::RANCell) =
    print(io, "RANCell(", size(ran.Wi, 2), " => ", size(ran.Wi, 1)÷3, ")")


struct RAN{M} <: AbstractRecurrentLayer
    cell::M
end

Flux.@layer :expand RAN

@doc raw"""
    RAN(input_size => hidden_size; kwargs...)

The `RANCell`, introduced in [this paper](https://arxiv.org/pdf/1705.07393), 
is a recurrent cell layer which provides additional memory through the
use of gates.

and returns both h_t anf c_t.

See [`RANCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
\tilde{c}_t &= W_c x_t, \\
i_t         &= \sigma(W_i x_t + U_i h_{t-1} + b_i), \\
f_t         &= \sigma(W_f x_t + U_f h_{t-1} + b_f), \\
c_t         &= i_t \odot \tilde{c}_t + f_t \odot c_{t-1}, \\
h_t         &= g(c_t)
\end{aligned}
```
"""
function RAN((input_size, hidden_size)::Pair; kwargs...)
    cell = RANCell(input_size => hidden_size; kwargs...)
    return RAN(cell)
end

function (ran::RAN)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    return scan(ran.cell, inp, state)
end

