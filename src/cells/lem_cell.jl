#https://arxiv.org/pdf/2110.04744
@doc raw"""
    LEMCell(input_size => hidden_size, [dt];
        init_kernel = glorot_uniform, init_recurrent_kernel = glorot_uniform,
        bias = true)

Long expressive memory unit [Rusch2022](@cite).
See [`LEM`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `dt`: timestep. Defaul is 1.0.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

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
struct LEMCell{I, H, Z, V, D} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    Wz::Z
    bias::V
    dt::D
end

@layer LEMCell

function LEMCell((input_size, hidden_size)::Pair{<:Int, <:Int}, dt::Number=1.0f0;
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size * 4, input_size)
    Wh = init_recurrent_kernel(hidden_size * 3, hidden_size)
    Wz = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))

    return LEMCell(Wi, Wh, Wz, b, eltype(Wi)(dt))
end

function (lem::LEMCell)(inp::AbstractVecOrMat, (state, z_state))
    _size_check(lem, inp, 1 => size(lem.Wi, 2))
    Wi, Wh, Wz, b = lem.Wi, lem.Wh, lem.Wz, lem.bias
    T = eltype(Wi)
    #split
    gxs = chunk(Wi * inp .+ b, 4; dims=1)
    ghs = chunk(Wh * state, 3; dims=1)
    gz = Wz * z_state

    msdt_bar = @. lem.dt * sigmoid_fast(gxs[1] + ghs[1])
    ms_dt = @. lem.dt * sigmoid_fast(gxs[2] + ghs[2])
    new_zstate = @. (T(1.0f0) - ms_dt) * z_state + ms_dt * tanh_fast(gxs[3] + ghs[3])
    new_state = @. (T(1.0f0) - msdt_bar) * state + msdt_bar * tanh_fast(gxs[4] + gz)
    return new_state, (new_state, new_zstate)
end

function Base.show(io::IO, lem::LEMCell)
    print(io, "LEMCell(", size(lem.Wi, 2), " => ", size(lem.Wi, 1) ÷ 4, ")")
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
    print(io, "LEM(", size(lem.cell.Wi, 2),
        " => ", size(lem.cell.Wi, 1) ÷ 4)
    print(io, ")")
end
