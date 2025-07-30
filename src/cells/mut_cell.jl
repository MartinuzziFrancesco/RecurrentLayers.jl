#https://proceedings.mlr.press/v37/jozefowicz15.pdf
@doc raw"""
    MUT1Cell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Mutated unit 1 cell [Jozefowicz2015](@cite).
See [`MUT1`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

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

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{z} \right), \\
    \mathbf{r}(t) &= \sigma\left( \mathbf{W}^{r}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{r}_{hh} \mathbf{h}(t) + \mathbf{b}^{r} \right), \\
    \mathbf{h}(t+1) &= \left[ \tanh\left( \mathbf{W}^{h}_{hh} \left(
        \mathbf{r}(t) \odot \mathbf{h}(t) \right) + \tanh\left(
        \mathbf{W}^{h}_{ih} \mathbf{x}(t) + \mathbf{b}^{h} \right) +
        \mathbf{b}^{h} \right) \right] \odot \mathbf{z}(t) \\
        &\quad + \mathbf{h}(t) \odot \left( 1 - \mathbf{z}(t) \right)
\end{aligned}
```

# Forward

    mutcell(inp, state)
    mutcell(inp)

## Arguments
- `inp`: The input to the mutcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the MUTCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct MUT1Cell{I,H,V,W,A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer MUT1Cell

function MUT1Cell((input_size, hidden_size)::Pair{<:Int,<:Int};
    init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
    bias::Bool=true, recurrent_bias::Bool=true,
    integration_mode::Symbol=:addition,
    independent_recurrence::Bool=false)
    weight_ih = init_kernel(3 * hidden_size, input_size)
    if independent_recurrence
        weight_hh = vec(init_recurrent_kernel(2 * hidden_size))
    else
        weight_hh = init_recurrent_kernel(2 * hidden_size, hidden_size)
    end
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    if integration_mode == :addition
        integration_fn = add_projections
    elseif integration_mode == :multiplicative_integration
        integration_fn = mul_projections
    else
        throw(ArgumentError(
            "integration_mode must be :addition or :multiplicative_integration; got $integration_mode"
        ))
    end
    return MUT1Cell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (mut::MUT1Cell)(inp::AbstractVecOrMat, state)
    _size_check(mut, inp, 1 => size(mut.weight_ih, 2))
    proj_ih = dense_proj(mut.weight_ih, inp, mut.bias_ih)
    gxs = chunk(proj_ih, 3; dims=1)
    whs = chunk(mut.weight_hh, 2; dims=1)
    bhs = chunk(mut.bias_hh, 2; dims=1)
    proj_hh_1 = dense_proj(whs[1], state, bhs[1])
    t_ones = eltype(Wi)(1.0f0)
    forget_gate = sigmoid_fast.(gxs[1])
    reset_gate = sigmoid_fast.(mut.integration_fn(gxs[2], proj_hh_1))
    proj_hh_2 = dense_proj(whs[2], (reset_gate .* state), bhs[2])
    candidate_state = tanh_fast.(mut.integration_fn(proj_hh_2, tanh_fast.(gxs[3])))
    new_state = candidate_state .* forget_gate .+ state .* (t_ones .- forget_gate)
    return new_state, new_state
end

function initialstates(mut::MUT1Cell)
    return zeros_like(mut.weight_hh, size(mut.weight_hh, 1) ÷ 2)
end

function Base.show(io::IO, mut::MUT1Cell)
    print(io, "MUT1Cell(", size(mut.weight_ih, 2), " => ", size(mut.weight_ih, 1) ÷ 3, ")")
end

@doc raw"""
    MUT1(input_size => hidden_size;
        return_state=false,
        kwargs...)

Mutated unit 1 network [Jozefowicz2015](@cite).
See [`MUT1Cell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

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
- `return_state`: Option to return the last state together with the output.
  Default is `false`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{z} \right), \\
    \mathbf{r}(t) &= \sigma\left( \mathbf{W}^{r}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{r}_{hh} \mathbf{h}(t) + \mathbf{b}^{r} \right), \\
    \mathbf{h}(t+1) &= \left[ \tanh\left( \mathbf{W}^{h}_{hh} \left(
        \mathbf{r}(t) \odot \mathbf{h}(t) \right) + \tanh\left(
        \mathbf{W}^{h}_{ih} \mathbf{x}(t) + \mathbf{b}^{h} \right) +
        \mathbf{b}^{h} \right) \right] \odot \mathbf{z}(t) \\
        &\quad + \mathbf{h}(t) \odot \left( 1 - \mathbf{z}(t) \right)
\end{aligned}
```

# Forward

    mut(inp, state)
    mut(inp)

## Arguments
- `inp`: The input to the mut. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the MUT. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct MUT1{S,M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand MUT1

function MUT1((input_size, hidden_size)::Pair{<:Int,<:Int};
    return_state::Bool=false, kwargs...)
    cell = MUT1Cell(input_size => hidden_size; kwargs...)
    return MUT1{return_state,typeof(cell)}(cell)
end

function functor(rnn::MUT1{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> MUT1{S,typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, mut::MUT1)
    print(io, "MUT1(", size(mut.cell.weight_ih, 2), " => ", size(mut.cell.weight_ih, 1))
    print(io, ")")
end

@doc raw"""
    MUT2Cell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Mutated unit 2 cell [Jozefowicz2015](@cite).
See [`MUT2`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

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

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{h}(t) + \mathbf{b}^{z} \right), \\
    \mathbf{r}(t) &= \sigma\left( \mathbf{x}(t) + \mathbf{W}^{r}_{hh}
        \mathbf{h}(t) + \mathbf{b}^{r} \right), \\
    \mathbf{h}(t+1) &= \left[ \tanh\left( \mathbf{W}^{h}_{hh} \left(
        \mathbf{r}(t) \odot \mathbf{h}(t) \right) + \mathbf{W}^{h}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{h} \right) \right] \odot \mathbf{z}(t) \\
        &\quad + \mathbf{h}(t) \odot \left( 1 - \mathbf{z}(t) \right)
\end{aligned}
```

# Forward

    mutcell(inp, state)
    mutcell(inp)

## Arguments
- `inp`: The input to the mutcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the MUTCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct MUT2Cell{I,H,V,W,A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer MUT2Cell

function MUT2Cell((input_size, hidden_size)::Pair{<:Int,<:Int};
    init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
    bias::Bool=true)
    bias::Bool=true, recurrent_bias::Bool=true,
    integration_mode::Symbol=:addition,
    independent_recurrence::Bool=false)
    weight_ih = init_kernel(3 * hidden_size, input_size)
    if independent_recurrence
        weight_hh = vec(init_recurrent_kernel(3 * hidden_size))
    else
        weight_hh = init_recurrent_kernel(3 * hidden_size, hidden_size)
    end
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    if integration_mode == :addition
        integration_fn = add_projections
    elseif integration_mode == :multiplicative_integration
        integration_fn = mul_projections
    else
        throw(ArgumentError(
            "integration_mode must be :addition or :multiplicative_integration; got $integration_mode"
        ))
    end
    return MUT2Cell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (mut::MUT2Cell)(inp::AbstractVecOrMat, state)
    _size_check(mut, inp, 1 => size(mut.weight_ih, 2))
    proj_ih = dense_proj(mut.weight_ih, inp, mut.bias_ih)
    gxs = chunk(proj_ih, 3; dims=1)
    whs = chunk(mut.weight_hh, 3; dims=1)
    bhs = chunk(mut.bias_hh, 3; dims=1)
    proj_hh_1 = dense_proj(whs[1], state, bhs[1])
    proj_hh_2 = dense_proj(whs[2], state, bhs[2])
    t_ones = eltype(weight_ih)(1.0f0)
    forget_gate = sigmoid_fast.(mut.integration_fn(gxs[1], proj_hh_1))
    # the dimensionlity alos does not work here like the paper describes it
    reset_gate = sigmoid_fast.(mut.integration_fn(gxs[2], proj_hh_2))
    proj_hh_3 = dense_proj(whs[3], (reset_gate .* state), bhs[3])
    candidate_state = tanh_fast.(mut.integration_fn(gxs[3], proj_hh_3))
    new_state = candidate_state .* forget_gate .+ state .* (t_ones .- forget_gate)
    return new_state, new_state
end

function Base.show(io::IO, mut::MUT2Cell)
    print(io, "MUT2Cell(", size(mut.weight_ih, 2), " => ", size(mut.weight_ih, 1) ÷ 3, ")")
end

@doc raw"""
    MUT2Cell(input_size => hidden_size;
        return_state=false,
        kwargs...)

Mutated unit 2 network [Jozefowicz2015](@cite).
See [`MUT2Cell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

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
- `return_state`: Option to return the last state together with the output.
  Default is `false`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{h}(t) + \mathbf{b}^{z} \right), \\
    \mathbf{r}(t) &= \sigma\left( \mathbf{x}(t) + \mathbf{W}^{r}_{hh}
        \mathbf{h}(t) + \mathbf{b}^{r} \right), \\
    \mathbf{h}(t+1) &= \left[ \tanh\left( \mathbf{W}^{h}_{hh} \left(
        \mathbf{r}(t) \odot \mathbf{h}(t) \right) + \mathbf{W}^{h}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{h} \right) \right] \odot \mathbf{z}(t) \\
        &\quad + \mathbf{h}(t) \odot \left( 1 - \mathbf{z}(t) \right)
\end{aligned}
```

# Forward

    mut(inp, state)
    mut(inp)

## Arguments
- `inp`: The input to the mut. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the MUT. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct MUT2{S,M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand MUT2

function MUT2((input_size, hidden_size)::Pair{<:Int,<:Int};
    return_state::Bool=false, kwargs...)
    cell = MUT2Cell(input_size => hidden_size; kwargs...)
    return MUT2{return_state,typeof(cell)}(cell)
end

function functor(rnn::MUT2{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> MUT2{S,typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, mut::MUT2)
    print(io, "MUT2(", size(mut.cell.weight_ih, 2), " => ", size(mut.cell.weight_ih, 1))
    print(io, ")")
end

@doc raw"""
    MUT3Cell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

Mutated unit 3 cell [Jozefowicz2015](@cite).
See [`MUT3`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

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


# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \tanh\left( \mathbf{h}(t) \right) + \mathbf{b}^{z}
        \right), \\
    \mathbf{r}(t) &= \sigma\left( \mathbf{W}^{r}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{r}_{hh} \mathbf{h}(t) + \mathbf{b}^{r} \right), \\
    \mathbf{h}(t+1) &= \left[ \tanh\left( \mathbf{W}^{h}_{hh} \left(
        \mathbf{r}(t) \odot \mathbf{h}(t) \right) + \mathbf{W}^{h}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{h} \right) \right] \odot \mathbf{z}(t) \\
        &\quad + \mathbf{h}(t) \odot \left( 1 - \mathbf{z}(t) \right)
\end{aligned}
```

# Forward

    mutcell(inp, state)
    mutcell(inp)

## Arguments
- `inp`: The input to the mutcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the MUTCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct MUT3Cell{I,H,V,W,A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer MUT3Cell

function MUT3Cell((input_size, hidden_size)::Pair{<:Int,<:Int};
    init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
    bias::Bool=true, recurrent_bias::Bool=true,
    integration_mode::Symbol=:addition,
    independent_recurrence::Bool=false)
    weight_ih = init_kernel(3 * hidden_size, input_size)
    if independent_recurrence
        weight_hh = vec(init_recurrent_kernel(3 * hidden_size))
    else
        weight_hh = init_recurrent_kernel(3 * hidden_size, hidden_size)
    end
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    if integration_mode == :addition
        integration_fn = add_projections
    elseif integration_mode == :multiplicative_integration
        integration_fn = mul_projections
    else
        throw(ArgumentError(
            "integration_mode must be :addition or :multiplicative_integration; got $integration_mode"
        ))
    end
    return MUT3Cell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (mut::MUT3Cell)(inp::AbstractVecOrMat, state)
    _size_check(mut, inp, 1 => size(mut.Wi, 2))
    Wi, Wh, b = mut.Wi, mut.Wh, mut.bias
    #split
    gxs = chunk(Wi * inp .+ b, 3; dims=1)
    ghs = chunk(Wh, 3; dims=1)

    t_ones = eltype(Wi)(1.0f0)
    forget_gate = sigmoid_fast.(gxs[1] .+ ghs[1] * tanh_fast(state))
    reset_gate = sigmoid_fast.(gxs[2] .+ ghs[2] * state)
    candidate_state = tanh_fast.(ghs[3] * (reset_gate .* state) .+ gxs[3])
    new_state = candidate_state .* forget_gate .+ state .* (t_ones .- forget_gate)
    return new_state, new_state
end

function Base.show(io::IO, mut::MUT3Cell)
    print(io, "MUT3Cell(", size(mut.Wi, 2), " => ", size(mut.Wi, 1) ÷ 3, ")")
end

@doc raw"""
    MUT3(input_size => hidden_size;
    return_state = false, kwargs...)

Mutated unit 3 network [Jozefowicz2015](@cite).
See [`MUT3Cell`](@ref) for a layer that processes a single sequence.

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
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \tanh\left( \mathbf{h}(t) \right) + \mathbf{b}^{z}
        \right), \\
    \mathbf{r}(t) &= \sigma\left( \mathbf{W}^{r}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{r}_{hh} \mathbf{h}(t) + \mathbf{b}^{r} \right), \\
    \mathbf{h}(t+1) &= \left[ \tanh\left( \mathbf{W}^{h}_{hh} \left(
        \mathbf{r}(t) \odot \mathbf{h}(t) \right) + \mathbf{W}^{h}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{h} \right) \right] \odot \mathbf{z}(t) \\
        &\quad + \mathbf{h}(t) \odot \left( 1 - \mathbf{z}(t) \right)
\end{aligned}
```

# Forward

    mut(inp, state)
    mut(inp)

## Arguments
- `inp`: The input to the mut. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the MUT. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct MUT3{S,M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand MUT3

function MUT3((input_size, hidden_size)::Pair{<:Int,<:Int};
    return_state::Bool=false, kwargs...)
    cell = MUT3Cell(input_size => hidden_size; kwargs...)
    return MUT3{return_state,typeof(cell)}(cell)
end

function functor(rnn::MUT3{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> MUT3{S,typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, mut::MUT3)
    print(io, "MUT3(", size(mut.cell.Wi, 2), " => ", size(mut.cell.Wi, 1))
    print(io, ")")
end
