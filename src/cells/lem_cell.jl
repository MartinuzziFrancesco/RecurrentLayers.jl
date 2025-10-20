#https://arxiv.org/pdf/2110.04744
@doc raw"""
    LEMCell(input_size => hidden_size, [dt];
        init_kernel = glorot_uniform, init_recurrent_kernel = glorot_uniform,
        init_cell_kernel = glorot_uniform,
        bias = true, recurrent_bias = true, cell_bias=true,
        independent_recurrence = false, integration_mode = :addition)

Long expressive memory unit [Rusch2022](@cite).
See [`LEM`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `dt`: timestep. Default is 1.0.

# Keyword arguments

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
    \boldsymbol{\Delta t}(t) &= \Delta \hat{t} \, \hat{\sigma} \left(
        \mathbf{W}^{1}_{hh} \mathbf{h}(t-1) + \mathbf{W}^{1}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{1} \right), \\
    \overline{\boldsymbol{\Delta t}}(t) &= \Delta \hat{t} \, \hat{\sigma}
        \left( \mathbf{W}^{2}_{hh} \mathbf{h}(t-1) + \mathbf{W}^{2}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{2} \right), \\
    \mathbf{z}(t) &= \left( 1 - \boldsymbol{\Delta t}(t) \right) \odot
        \mathbf{z}(t-1) + \boldsymbol{\Delta t}(t) \odot \sigma \left(
        \mathbf{W}^{z}_{hh} \mathbf{h}(t-1) + \mathbf{W}^{z}_{ih} \mathbf{x}(t)
        + \mathbf{b}^{z} \right), \\
    \mathbf{h}(t) &= \left( 1 - \boldsymbol{\Delta t}(t) \right) \odot
        \mathbf{h}(t-1) + \boldsymbol{\Delta t}(t) \odot \sigma \left(
        \mathbf{W}^{h}_{zh} \mathbf{z}(t) + \mathbf{W}^{h}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{h} \right)
\end{aligned}
```

# Forward

    lemcell(inp, (state, cstate))
    lemcell(inp)

## Arguments
- `inp`: The input to the lemcell. It should be a vector of size `input_size`
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
struct LEMCell{I, H, Z, V, W, C, A, D} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    weight_ch::Z
    bias_ih::V
    bias_hh::W
    bias_ch::C
    integration_fn::A
    dt::D
end

@layer LEMCell

function LEMCell((input_size, hidden_size)::Pair{<:Int, <:Int}, dt::Number=1.0f0;
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        init_cell_kernel=glorot_uniform, bias::Bool=true, recurrent_bias::Bool=true,
        cell_bias::Bool=true, integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(hidden_size * 4, input_size)
    if independent_recurrence
        weight_hh = vec(init_recurrent_kernel(hidden_size * 3))
    else
        weight_hh = init_recurrent_kernel(hidden_size * 3, hidden_size)
    end
    weight_ch = init_cell_kernel(hidden_size, hidden_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    bias_ch = create_bias(weight_ch, cell_bias, size(weight_ch, 1))
    integration_fn = _integration_fn(integration_mode)
    return LEMCell(weight_ih, weight_hh, weight_ch, bias_ih, bias_hh, bias_ch,
        integration_fn, eltype(weight_ih)(dt))
end

function (lem::LEMCell)(inp::AbstractVecOrMat, (state, z_state))
    _size_check(lem, inp, 1 => size(lem.weight_ih, 2))
    T = eltype(lem.weight_ih)
    proj_ih = dense_proj(lem.weight_ih, inp, lem.bias_ih)
    proj_hh = dense_proj(lem.weight_hh, state, lem.bias_hh)
    proj_ch = dense_proj(lem.weight_ch, z_state, lem.bias_ch) #gz
    gxs = chunk(proj_ih, 4; dims=1)
    ghs = chunk(proj_hh, 3; dims=1)
    int_proj_3 = lem.integration_fn(gxs[3], ghs[3])
    int_proj_4 = lem.integration_fn(gxs[4], proj_ch)
    msdt_bar = lem.dt .* sigmoid_fast.(lem.integration_fn(gxs[1], ghs[1]))
    ms_dt = lem.dt .* sigmoid_fast.(lem.integration_fn(gxs[2], ghs[2]))
    new_zstate = @. (T(1.0f0) - ms_dt) * z_state + ms_dt * tanh_fast(int_proj_3)
    new_state = @. (T(1.0f0) - msdt_bar) * state + msdt_bar * tanh_fast(int_proj_4)
    return new_state, (new_state, new_zstate)
end

function initialstates(lem::LEMCell)
    state = zeros_like(lem.weight_hh, size(lem.weight_hh, 1) ÷ 3)
    second_state = zeros_like(lem.weight_hh, size(lem.weight_hh, 1) ÷ 3)
    return state, second_state
end

function Base.show(io::IO, lem::LEMCell)
    print(io, "LEMCell(", size(lem.weight_ih, 2), " => ", size(lem.weight_ih, 1) ÷ 4, ")")
end

@doc raw"""
    LEM(input_size => hidden_size, [dt];
        return_state=false, init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform, bias = true)

Long expressive memory network [Rusch2022](@cite).
See [`LEMCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `dt`: timestep. Defaul is 1.0.

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
    \boldsymbol{\Delta t}(t) &= \Delta \hat{t} \, \hat{\sigma} \left(
        \mathbf{W}^{1}_{hh} \mathbf{h}(t-1) + \mathbf{W}^{1}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{1} \right), \\
    \overline{\boldsymbol{\Delta t}}(t) &= \Delta \hat{t} \, \hat{\sigma}
        \left( \mathbf{W}^{2}_{hh} \mathbf{h}(t-1) + \mathbf{W}^{2}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{2} \right), \\
    \mathbf{z}(t) &= \left( 1 - \boldsymbol{\Delta t}(t) \right) \odot
        \mathbf{z}(t-1) + \boldsymbol{\Delta t}(t) \odot \sigma \left(
        \mathbf{W}^{z}_{hh} \mathbf{h}(t-1) + \mathbf{W}^{z}_{ih} \mathbf{x}(t)
        + \mathbf{b}^{z} \right), \\
    \mathbf{h}(t) &= \left( 1 - \boldsymbol{\Delta t}(t) \right) \odot
        \mathbf{h}(t-1) + \boldsymbol{\Delta t}(t) \odot \sigma \left(
        \mathbf{W}^{h}_{zh} \mathbf{z}(t) + \mathbf{W}^{h}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{h} \right)
\end{aligned}
```

# Forward

    lem(inp, (state, zstate))
    lem(inp)

## Arguments
- `inp`: The input to the LEM. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the LEM.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct LEM{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand LEM

function LEM((input_size, hidden_size)::Pair{<:Int, <:Int}, dt::Number=1.0;
        return_state::Bool=false, kwargs...)
    cell = LEMCell(input_size => hidden_size, dt; kwargs...)
    return LEM{return_state, typeof(cell)}(cell)
end

function functor(rnn::LEM{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> LEM{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, lem::LEM)
    print(io, "LEM(", size(lem.cell.weight_ih, 2),
        " => ", size(lem.cell.weight_ih, 1) ÷ 4)
    print(io, ")")
end
