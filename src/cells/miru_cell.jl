
@doc raw"""
    MiRU1Cell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition,
        update_coefficient = 0.5)

Minion gated recurrent unit 1 [Zyarah2026](@cite).
See [`MiRU1`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: activation function. Default `tanh_fast`.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `update_coefficient`: controls the dynamic update of the hidden states.
  Default is 0.5.

# Equations

```math
\begin{aligned}
    \mathbf{r}(t) &= \sigma\!\left(
        \mathbf{W}_{r}\mathbf{x}(t) + \mathbf{b}_{r}
        + \mathbf{U}_{r}\mathbf{h}(t-1)
    \right), \\
    \tilde{\mathbf{h}}(t) &= \tanh\!\left(
        \mathbf{W}_{h}\mathbf{x}(t) + \mathbf{b}_{h}
        + \mathbf{U}_{h}\!\left(\mathbf{r}(t) \odot \mathbf{h}(t-1)\right)
    \right), \\
    \mathbf{h}(t) &= \boldsymbol{\lambda} \odot \mathbf{h}(t-1)
        + \left(1 - \boldsymbol{\lambda}\right) \odot \tilde{\mathbf{h}}(t)
\end{aligned}
```

# Forward

    mirucell(inp, (state, cstate))
    mirucell(inp)

## Arguments
- `inp`: The input to the mirucell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the MiRUCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state.
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct MiRU1Cell{I, H, V, W, L, F, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    update_coefficient::L
    activation::F
    integration_fn::A
end

function MiRU1Cell((input_size, hidden_size)::Pair{<:Int, <:Int}, activation=tanh_fast;
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        update_coefficient::AbstractFloat=0.5,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(2 * hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 2)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    _update_coefficient = eltype(weight_ih)(update_coefficient)
    integration_fn = _integration_fn(integration_mode)

    return MiRU1Cell(weight_ih, weight_hh, bias_ih, bias_hh,
        _update_coefficient, activation, integration_fn)
end

function (miru::MiRU1Cell)(inp::AbstractVecOrMat, state)
    _size_check(miru, inp, 1 => size(miru.weight_ih, 2))
    proj_ih = dense_proj(miru.weight_ih, inp, miru.bias_ih)
    gxs = chunk(proj_ih, 2; dims=1)
    chunk_whh = chunk(miru.weight_hh, 2; dims=1)
    chunk_bhh = chunk(miru.bias_hh, 2; dims=1)
    rec_r = dense_proj(chunk_whh[1], state, chunk_bhh[1])
    first_gate = miru.activation.(miru.integration_fn(gxs[1], rec_r))
    rec_h = dense_proj(chunk_whh[2], first_gate .* state, chunk_bhh[2])
    candidate_state = tanh_fast.(miru.integration_fn(gxs[2], rec_h))
    new_state = miru.update_coefficient .* state .+
                (1 - miru.update_coefficient) .* candidate_state

    return new_state, new_state
end

function initialstates(miru::MiRU1Cell)
    state = zeros_like(miru.weight_hh, size(miru.weight_hh, 1) ÷ 2)
    return state
end

function Base.show(io::IO, miru::MiRU1Cell)
    print(io, "MiRU1Cell(", size(miru.weight_ih, 2), " => ", size(miru.weight_ih, 1) ÷ 2)
    print(io, ")")
end

@doc raw"""
    MiRU1(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition,
        update_coefficient = 0.5)

Minion gated recurrent unit 1 [Zyarah2026](@cite).
See [`MiRU1Cell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: activation function. Default `tanh_fast`.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `update_coefficient`: controls the dynamic update of the hidden states.
  Default is 0.5.

# Equations

```math
\begin{aligned}
    \mathbf{r}(t) &= \sigma\!\left(
        \mathbf{W}_{r}\mathbf{x}(t) + \mathbf{b}_{r}
        + \mathbf{U}_{r}\mathbf{h}(t-1)
    \right), \\
    \tilde{\mathbf{h}}(t) &= \tanh\!\left(
        \mathbf{W}_{h}\mathbf{x}(t) + \mathbf{b}_{h}
        + \mathbf{U}_{h}\!\left(\mathbf{r}(t) \odot \mathbf{h}(t-1)\right)
    \right), \\
    \mathbf{h}(t) &= \boldsymbol{\lambda} \odot \mathbf{h}(t-1)
        + \left(1 - \boldsymbol{\lambda}\right) \odot \tilde{\mathbf{h}}(t)
\end{aligned}
```

# Forward

    miru(inp, (state, cstate))
    miru(inp)

## Arguments
- `inp`: The input to the miru. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the MiRU1.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct MiRU1{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand MiRU1

function MiRU1((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = MiRU1Cell(input_size => hidden_size; kwargs...)
    return MiRU1{return_state, typeof(cell)}(cell)
end

function functor(miru::MiRU1{S}) where {S}
    params = (cell=miru.cell,)
    reconstruct = p -> MiRU1{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, miru::MiRU1)
    print(
        io, "MiRU1(", size(miru.cell.weight_ih, 2), " => ", size(miru.cell.weight_ih, 1))
    print(io, ")")
end

@doc raw"""
    MiRU2Cell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition,
        update_coefficient = 0.5, reset_coefficient = 0.5)

Minion gated recurrent unit 2 [Zyarah2026](@cite).
See [`MiRU2`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: activation function. Default `tanh_fast`.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `update_coefficient`: controls the dynamic update of the hidden states.
  Default is 0.5.
- `reset_coefficient`: determines how much of the previous hidden state should
  be forgotten or reset before combining it with the new input.
  Default is 0.5.

# Equations

```math
\begin{aligned}
    \tilde{\mathbf{h}}(t) &= \tanh\!\left(
        \mathbf{W}_{h}\mathbf{x}(t) + \mathbf{b}_{h}
        + \mathbf{U}_{h}\!\left(\boldsymbol{\theta} \odot \mathbf{h}(t-1)\right)
    \right), \\
    \mathbf{h}(t) &= \boldsymbol{\lambda} \odot \mathbf{h}(t-1)
        + \left(1 - \boldsymbol{\lambda}\right) \odot \tilde{\mathbf{h}}(t)
\end{aligned}
```

# Forward

    mirucell(inp, (state, cstate))
    mirucell(inp)

## Arguments
- `inp`: The input to the mirucell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the MiRUCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state.
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct MiRU2Cell{I, H, V, W, L, R, F, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    update_coefficient::L
    reset_coefficient::R
    activation::F
    integration_fn::A
end

function MiRU2Cell((input_size, hidden_size)::Pair{<:Int, <:Int}, activation=tanh;
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        update_coefficient::AbstractFloat=0.5,
        reset_coefficient::AbstractFloat=0.5,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    _update_coefficient = eltype(weight_ih)(update_coefficient)
    _reset_coefficient = eltype(weight_ih)(reset_coefficient)
    integration_fn = _integration_fn(integration_mode)

    return MiRU2Cell(weight_ih, weight_hh, bias_ih, bias_hh, _update_coefficient,
        _reset_coefficient, activation, integration_fn)
end

function (miru::MiRU2Cell)(inp::AbstractVecOrMat, state)
    _size_check(miru, inp, 1 => size(miru.weight_ih, 2))
    proj_ih = dense_proj(miru.weight_ih, inp, miru.bias_ih)
    proj_hh = dense_proj(miru.weight_hh, miru.reset_coefficient .* state, miru.bias_hh)
    candidate_state = miru.activation.(miru.integration_fn(proj_ih, proj_hh))
    new_state = miru.update_coefficient .* state +
                (1 .- miru.update_coefficient) .* candidate_state

    return new_state, new_state
end

function initialstates(miru::MiRU2Cell)
    state = zeros_like(miru.weight_hh, size(miru.weight_hh, 1))
    return state
end

function Base.show(io::IO, miru::MiRU2Cell)
    print(io, "MiRU2Cell(", size(miru.weight_ih, 2), " => ", size(miru.weight_ih, 1))
    print(io, ")")
end

@doc raw"""
    MiRU2(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition,
        update_coefficient = 0.5, reset_coefficient = 0.5)

Minion gated recurrent unit 2 [Zyarah2026](@cite).
See [`MiRU2`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: activation function. Default `tanh_fast`.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `update_coefficient`: controls the dynamic update of the hidden states.
  Default is 0.5.
- `reset_coefficient`: determines how much of the previous hidden state should
  be forgotten or reset before combining it with the new input.
  Default is 0.5.

# Equations

```math
\begin{aligned}
    \tilde{\mathbf{h}}(t) &= \tanh\!\left(
        \mathbf{W}_{h}\mathbf{x}(t) + \mathbf{b}_{h}
        + \mathbf{U}_{h}\!\left(\boldsymbol{\theta} \odot \mathbf{h}(t-1)\right)
    \right), \\
    \mathbf{h}(t) &= \boldsymbol{\lambda} \odot \mathbf{h}(t-1)
        + \left(1 - \boldsymbol{\lambda}\right) \odot \tilde{\mathbf{h}}(t)
\end{aligned}
```

# Forward

    miru(inp, (state, cstate))
    miru(inp)

## Arguments
- `inp`: The input to the miru. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the MiRU2.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct MiRU2{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand MiRU2

function MiRU2((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = MiRU2Cell(input_size => hidden_size; kwargs...)
    return MiRU2{return_state, typeof(cell)}(cell)
end

function functor(miru::MiRU2{S}) where {S}
    params = (cell=miru.cell,)
    reconstruct = p -> MiRU2{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, miru::MiRU2)
    print(
        io, "MiRU2(", size(miru.cell.weight_ih, 2), " => ", size(miru.cell.weight_ih, 1))
    print(io, ")")
end
