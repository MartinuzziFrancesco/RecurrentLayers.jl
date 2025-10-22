#https://arxiv.org/abs/2010.00951
@doc raw"""
    coRNNCell(input_size => hidden_size, [dt];
        gamma=0.0, epsilon=0.0,
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        init_cell_kernel = glorot_uniform,
        bias = true, recurrent_bias = true, cell_bias=true,
        independent_recurrence = false, integration_mode = :addition)

Coupled oscillatory recurrent neural unit [Rusch2021](@cite).
See [`coRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `dt`: time step. Default is 1.0.

# Keyword arguments

- `gamma`: damping for state. Default is 0.0.
- `epsilon`: damping for candidate state. Default is 0.0.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_cell_kernel`: initializer for the cell to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `cell_bias`: include cell to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.


# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \mathbf{z}(t-1) + \Delta t \, \sigma \left( \mathbf{W}_{hh}
        \mathbf{h}(t-1) + \mathbf{W}_{zh} \mathbf{z}(t-1) + \mathbf{W}_{ih}
        \mathbf{x}(t) + \mathbf{b} \right) - \Delta t \, \gamma \mathbf{h}(t-1)
        - \Delta t \, \epsilon \mathbf{z}(t), \\
    \mathbf{h}(t) &= \mathbf{h}(t-1) + \Delta t \, \mathbf{z}(t),
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
struct coRNNCell{I, H, Z, V, W, C, A, D, G, E} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    weight_ch::Z
    bias_ih::V
    bias_hh::W
    bias_ch::C
    integration_fn::A
    dt::D
    gamma::G
    epsilon::E
end

@layer coRNNCell

function coRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int},
        dt::Number=1.0f0; gamma::Number=0.0f0, epsilon::Number=0.0f0,
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        init_cell_kernel=glorot_uniform, bias::Bool=true, recurrent_bias::Bool=true,
        cell_bias::Bool=true, integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size)
    weight_ch = init_cell_kernel(hidden_size, hidden_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    bias_ch = create_bias(weight_ch, cell_bias, size(weight_ch, 1))
    T = eltype(weight_ih)
    integration_fn = _integration_fn(integration_mode)
    return coRNNCell(weight_ih, weight_hh, weight_ch, bias_ih, bias_hh, bias_ch,
        integration_fn, T(dt), T(gamma), T(epsilon))
end

function (cornn::coRNNCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(cornn, inp, 1 => size(cornn.weight_ih, 2))
    proj_ih = dense_proj(cornn.weight_ih, inp, cornn.bias_ih)
    proj_hh = dense_proj(cornn.weight_hh, state, cornn.bias_hh)
    integrated_proj = cornn.integration_fn(proj_ih, proj_hh)
    preact = integrated_proj .+ cornn.weight_ch * c_state .+ cornn.bias_ch
    new_cstate = c_state .+ cornn.dt .* tanh_fast.(preact) .-
                 cornn.dt .* cornn.gamma .* state .-
                 cornn.dt .* cornn.epsilon .* c_state
    new_state = @. state + cornn.dt * new_cstate
    return new_state, (new_state, new_cstate)
end

function initialstates(cornn::coRNNCell)
    state = zeros_like(cornn.weight_hh, size(cornn.weight_hh, 1))
    second_state = zeros_like(cornn.weight_hh, size(cornn.weight_hh, 1))
    return state, second_state
end

function Base.show(io::IO, cornn::coRNNCell)
    print(io, "coRNNCell(", size(cornn.weight_ih, 2), " => ", size(cornn.weight_ih, 1), ")")
end

@doc raw"""
    coRNN(input_size => hidden_size, [dt];
        gamma=0.0, epsilon=0.0,
        return_state=false, init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        init_cell_kernel = glorot_uniform,
        bias = true, recurrent_bias = true, cell_bias=true
        independent_recurrence = false, integration_mode = :addition)

Coupled oscillatory recurrent neural unit [Rusch2021](@cite).
See [`coRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `dt`: time step. Default is 1.0.

# Keyword arguments

- `gamma`: damping for state. Default is 0.0.
- `epsilon`: damping for candidate state. Default is 0.0.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_cell_kernel`: initializer for the cell to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `cell_bias`: include cell to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `return_state`: Option to return the last state together with the output.
  Default is `false`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \mathbf{z}(t-1) + \Delta t \, \sigma \left( \mathbf{W}_{hh}
        \mathbf{h}(t-1) + \mathbf{W}_{zh} \mathbf{z}(t-1) + \mathbf{W}_{ih}
        \mathbf{x}(t) + \mathbf{b} \right) - \Delta t \, \gamma \mathbf{h}(t-1)
        - \Delta t \, \epsilon \mathbf{z}(t), \\
    \mathbf{h}(t) &= \mathbf{h}(t-1) + \Delta t \, \mathbf{z}(t),
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
    print(io, "coRNN(", size(cornn.cell.weight_ih, 2),
        " => ", size(cornn.cell.weight_ih, 1))
    print(io, ")")
end
