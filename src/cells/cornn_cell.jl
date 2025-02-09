#https://arxiv.org/abs/2010.00951
@doc raw"""
    coRNNCell(input_size => hidden_size, [dt];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Coupled oscillatory recurrent neural unit](https://arxiv.org/abs/2010.00951).
See [`coRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `dt`: time step. Default is 1.0

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
\mathbf{y}_n &= y_{n-1} + \Delta t \mathbf{z}_n, \\
\mathbf{z}_n &= z_{n-1} + \Delta t \sigma \left( \mathbf{W} y_{n-1} +
    \mathcal{W} z_{n-1} + \mathbf{V} u_n + \mathbf{b} \right) -
    \Delta t \gamma y_{n-1} - \Delta t \epsilon \mathbf{z}_n,
\end{aligned}
```

# Forward

    cornncell(inp, (state, cstate))
    cornncell(inp)

## Arguments
- `inp`: The input to the cornncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the RANCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state. 
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct coRNNCell{I, H, Z, V, D} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    Wz::Z
    bias::V
    dt::D
end

@layer coRNNCell

function coRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int},
        gamma::Number, epsilon::Number, dt::Number=1.0;
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size, hidden_size)
    Wz = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))

    return coRNNCell(Wi, Wh, Wz, b, eltype(Wi)(dt))
end

function (cornn::coRNNCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(cornn, inp, 1 => size(cornn.Wi, 2))
    Wi, Wh, Wz, b = cornn.Wi, cornn.Wh, cornn.Wz, cornn.bias
    dt, gamma, epsilon = cornn.dt, cornn.gamma, cornn.epsilon
    new_cstate = c_state .+ dt .* tanh_fast.(Wi * inp .+ Wh * state .+
        Wz * c_state .+ b) .- dt .* gamma .* state .- dt .* epsilon .* c_state
    new_state = state .+ dt .* new_cstate
    return new_state, (new_state, new_cstate)
end

function Base.show(io::IO, cornn::coRNNCell)
    print(io, "coRNNCell(", size(cornn.Wi, 2), " => ", size(cornn.Wi, 1), ")")
end