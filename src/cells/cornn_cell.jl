#https://arxiv.org/abs/2010.00951
@doc raw"""
    coRNNCell(input_size => hidden_size, [dt];
        gamma=0.0, epsilon=0.0,
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Coupled oscillatory recurrent neural unit](https://arxiv.org/abs/2010.00951).
See [`coRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `dt`: time step. Default is 1.0.

# Keyword arguments

- `gamma`: damping for state. Default is 0.0.
- `epsilon`: damping for candidate state. Default is 0.0.
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
- `(state, cstate)`: A tuple containing the hidden and cell states of the coRNNCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state. 
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct coRNNCell{I, H, Z, V, D, G, E} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    Wz::Z
    bias::V
    dt::D
    gamma::G
    epsilon::E
end

@layer coRNNCell

function coRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int},
        dt::Number=1.0f0; gamma::Number=0.0f0, epsilon::Number=0.0f0,
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size, hidden_size)
    Wz = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    T = eltype(Wi)

    return coRNNCell(Wi, Wh, Wz, b, T(dt), T(gamma), T(epsilon))
end

function (cornn::coRNNCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(cornn, inp, 1 => size(cornn.Wi, 2))
    Wi, Wh, Wz, b = cornn.Wi, cornn.Wh, cornn.Wz, cornn.bias
    dt, gamma, epsilon = cornn.dt, cornn.gamma, cornn.epsilon
    new_cstate = c_state .+ dt .* tanh_fast.(Wi * inp .+ Wh * state .+
                                             Wz * c_state .+ b) .- dt .* gamma .* state .-
                 dt .* epsilon .* c_state
    new_state = state .+ dt .* new_cstate
    return new_state, (new_state, new_cstate)
end

function Base.show(io::IO, cornn::coRNNCell)
    print(io, "coRNNCell(", size(cornn.Wi, 2), " => ", size(cornn.Wi, 1), ")")
end

@doc raw"""
    coRNN(input_size => hidden_size, [dt];
        gamma=0.0, epsilon=0.0,
        return_state=false, init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform, bias = true)

[Coupled oscillatory recurrent neural unit](https://arxiv.org/abs/2010.00951).
See [`coRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `dt`: time step. Default is 1.0.

# Keyword arguments

- `gamma`: damping for state. Default is 0.0.
- `epsilon`: damping for candidate state. Default is 0.0.
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`
- `return_state`: Option to return the last state together with the output.
  Default is `false`.
  
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

    cornn(inp, (state, zstate))
    cornn(inp)

## Arguments
- `inp`: The input to the `cornn`. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the `coRNN`. 
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct coRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand coRNN

function coRNN((input_size, hidden_size)::Pair{<:Int, <:Int}, args...;
        return_state::Bool=false, kwargs...)
    cell = coRNNCell(input_size => hidden_size, args...; kwargs...)
    return coRNN{return_state, typeof(cell)}(cell)
end

function functor(cornn::coRNN{S}) where {S}
    params = (cell=cornn.cell,)
    reconstruct = p -> coRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, cornn::coRNN)
    print(io, "coRNN(", size(cornn.cell.Wi, 2),
        " => ", size(cornn.cell.Wi, 1))
    print(io, ")")
end
