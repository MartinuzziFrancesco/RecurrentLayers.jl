#https://arxiv.org/abs/2212.00228
@doc raw"""
    TauGRUCell(input_size => hidden_size;
        delay = 1,
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Gated recurrent unit with weighted time-delay feedback [Erichson2025](@cite).
See [`TauGRU`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `delay`: positive integer recurrent delay. Default is `1`.
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
    \mathbf{u}_n &= \tanh\left( \mathbf{W}_1 \mathbf{h}_n +
        \mathbf{U}_1 \mathbf{x}_n \right), \\
    \mathbf{z}_n &= \tanh\left( \mathbf{W}_2 \mathbf{h}_{n-d} +
        \mathbf{U}_2 \mathbf{x}_n \right), \\
    \mathbf{g}_n &= \sigma\left( \mathbf{W}_3 \mathbf{h}_n +
        \mathbf{U}_3 \mathbf{x}_n \right), \\
    \mathbf{a}_n &= \sigma\left( \mathbf{W}_4 \mathbf{h}_n +
        \mathbf{U}_4 \mathbf{x}_n \right), \\
    \mathbf{h}_{n+1} &= \left(1 - \mathbf{g}_n\right) \odot \mathbf{h}_n +
        \mathbf{g}_n \odot \left(\mathbf{u}_n +
        \mathbf{a}_n \odot \mathbf{z}_n\right)
\end{aligned}
```

# Forward

    taugrucell(inp, state)
    taugrucell(inp)

## Arguments
- `inp`: The input to the taugrucell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: A tuple `(hidden_state, history)`. The hidden state should be a vector
  of size `hidden_size` or a matrix of size `hidden_size x batch_size`. The history is
  a flattened delay buffer of size `hidden_size * delay` or
  `hidden_size * delay x batch_size`. If not provided, both are initialized to zeros
  by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_history)` is the new hidden state together with the shifted
  delay buffer.
"""
struct TauGRUCell{I, H, V, W, A} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
    delay::Int
end

@layer TauGRUCell

function TauGRUCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        delay::Int=1,
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    delay > 0 || throw(ArgumentError("delay must be a positive integer; got $delay"))
    weight_ih = init_kernel(4 * hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 4)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)
    return TauGRUCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn, delay)
end

function _taugru_batch_state(state::AbstractVector, inp::AbstractMatrix)
    return repeat(state, 1, size(inp, 2))
end

_taugru_batch_state(state::AbstractVecOrMat, inp::AbstractVector) = state
_taugru_batch_state(state::AbstractMatrix, inp::AbstractMatrix) = state

function _taugru_delayed_state(history::AbstractVector, hidden_size::Int)
    return history[1:hidden_size]
end

function _taugru_delayed_state(history::AbstractMatrix, hidden_size::Int)
    return history[1:hidden_size, :]
end

function _taugru_next_history(history::AbstractVector, state::AbstractVector, hidden_size::Int)
    if length(history) == hidden_size
        return state
    else
        return vcat(history[(hidden_size + 1):end], state)
    end
end

function _taugru_next_history(history::AbstractMatrix, state::AbstractMatrix, hidden_size::Int)
    if size(history, 1) == hidden_size
        return state
    else
        return vcat(history[(hidden_size + 1):end, :], state)
    end
end

function (taugru::TauGRUCell)(inp::AbstractVecOrMat, (state, history))
    _size_check(taugru, inp, 1 => size(taugru.weight_ih, 2))
    hidden_size = size(taugru.weight_ih, 1) ÷ 4
    state = _taugru_batch_state(state, inp)
    history = _taugru_batch_state(history, inp)
    delayed_state = _taugru_delayed_state(history, hidden_size)

    proj_ih = dense_proj(taugru.weight_ih, inp, taugru.bias_ih)
    gxs = chunk(proj_ih, 4; dims=1)
    ghs = chunk(taugru.weight_hh, 4; dims=1)
    bhs = _chunked_bias(taugru.bias_hh, 4)
    t_ones = eltype(taugru.weight_ih)(1.0f0)

    proj_u = dense_proj(ghs[1], state, bhs[1])
    proj_z = dense_proj(ghs[2], delayed_state, bhs[2])
    proj_g = dense_proj(ghs[3], state, bhs[3])
    proj_a = dense_proj(ghs[4], state, bhs[4])

    candidate_state = tanh_fast.(taugru.integration_fn(gxs[1], proj_u))
    delayed_candidate = tanh_fast.(taugru.integration_fn(gxs[2], proj_z))
    update_gate = sigmoid_fast.(taugru.integration_fn(gxs[3], proj_g))
    feedback_gate = sigmoid_fast.(taugru.integration_fn(gxs[4], proj_a))
    new_state = @. (t_ones - update_gate) * state +
                   update_gate * (candidate_state + feedback_gate * delayed_candidate)
    new_history = _taugru_next_history(history, state, hidden_size)
    return new_state, (new_state, new_history)
end

function initialstates(taugru::TauGRUCell)
    hidden_size = size(taugru.weight_ih, 1) ÷ 4
    initial_state = zeros_like(taugru.weight_hh, hidden_size)
    initial_history = zeros_like(taugru.weight_hh, hidden_size * taugru.delay)
    return initial_state, initial_history
end

function Base.show(io::IO, taugru::TauGRUCell)
    print(io, "TauGRUCell(", size(taugru.weight_ih, 2), " => ",
        size(taugru.weight_ih, 1) ÷ 4)
    if taugru.delay != 1
        print(io, ", delay=", taugru.delay)
    end
    print(io, ")")
end

@doc raw"""
    TauGRU(input_size => hidden_size;
        return_state = false, kwargs...)

Gated recurrent unit with weighted time-delay feedback [Erichson2025](@cite).
See [`TauGRUCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `return_state`: Option to return the last state together with the output.
  Default is `false`.
- `delay`: positive integer recurrent delay. Default is `1`.
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
    \mathbf{u}_n &= \tanh\left( \mathbf{W}_1 \mathbf{h}_n +
        \mathbf{U}_1 \mathbf{x}_n \right), \\
    \mathbf{z}_n &= \tanh\left( \mathbf{W}_2 \mathbf{h}_{n-d} +
        \mathbf{U}_2 \mathbf{x}_n \right), \\
    \mathbf{g}_n &= \sigma\left( \mathbf{W}_3 \mathbf{h}_n +
        \mathbf{U}_3 \mathbf{x}_n \right), \\
    \mathbf{a}_n &= \sigma\left( \mathbf{W}_4 \mathbf{h}_n +
        \mathbf{U}_4 \mathbf{x}_n \right), \\
    \mathbf{h}_{n+1} &= \left(1 - \mathbf{g}_n\right) \odot \mathbf{h}_n +
        \mathbf{g}_n \odot \left(\mathbf{u}_n +
        \mathbf{a}_n \odot \mathbf{z}_n\right)
\end{aligned}
```

# Forward

    taugru(inp, state)
    taugru(inp)

## Arguments
- `inp`: The input to the taugru. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: A tuple `(hidden_state, history)`. If not provided, both are initialized to
  zeros by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden states `new_states` and
  the last state of the iteration.
"""
struct TauGRU{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand TauGRU

function TauGRU((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = TauGRUCell(input_size => hidden_size; kwargs...)
    return TauGRU{return_state, typeof(cell)}(cell)
end

function functor(taugru::TauGRU{S}) where {S}
    params = (cell=taugru.cell,)
    reconstruct = p -> TauGRU{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, taugru::TauGRU)
    print(io, "TauGRU(", size(taugru.cell.weight_ih, 2), " => ",
        size(taugru.cell.weight_ih, 1) ÷ 4)
    if taugru.cell.delay != 1
        print(io, ", delay=", taugru.cell.delay)
    end
    print(io, ")")
end
