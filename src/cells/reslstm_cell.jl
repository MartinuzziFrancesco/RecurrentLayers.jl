#https://arxiv.org/abs/1701.03360
@doc raw"""
    ResLSTMCell(input_size => hidden_size;
        memory_size = hidden_size,
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        init_peephole_kernel = glorot_uniform,
        init_projection_kernel = glorot_uniform,
        init_residual_kernel = glorot_uniform,
        bias = true, recurrent_bias = true, peephole_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Residual long short-term memory cell [Kim2017](@cite).
See [`ResLSTM`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and output dimension of the layer.

# Keyword arguments

- `memory_size`: internal memory cell dimension. Default is `hidden_size`.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_peephole_kernel`: initializer for the peephole weights.
    Default is `glorot_uniform`.
- `init_projection_kernel`: initializer for the cell output projection weights.
    Default is `glorot_uniform`.
- `init_residual_kernel`: initializer for the residual projection weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `peephole_bias`: include peephole to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.

# Equations

```math
\begin{aligned}
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{hh} \mathbf{h}(t-1) + \mathbf{p}^{i} \odot
        \mathbf{c}(t-1) + \mathbf{b}^{i} \right), \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{p}^{f} \odot
        \mathbf{c}(t-1) + \mathbf{b}^{f} \right), \\
    \mathbf{g}(t) &= \tanh\left( \mathbf{W}^{g}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{g}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{g} \right), \\
    \mathbf{c}(t) &= \mathbf{f}(t) \odot \mathbf{c}(t-1) +
        \mathbf{i}(t) \odot \mathbf{g}(t), \\
    \mathbf{o}(t) &= \sigma\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{hh} \mathbf{h}(t-1) + \mathbf{W}^{o}_{ch}
        \mathbf{c}(t) + \mathbf{b}^{o} \right), \\
    \mathbf{m}(t) &= \mathbf{W}_{p} \tanh\left( \mathbf{c}(t) \right), \\
    \mathbf{h}(t) &= \mathbf{o}(t) \odot
        \left( \mathbf{m}(t) + \mathbf{W}_{r} \mathbf{x}(t) \right)
\end{aligned}
```

# Forward

    reslstmcell(inp, (state, cstate))
    reslstmcell(inp)

## Arguments

- `inp`: The input to the reslstmcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the ResLSTMCell.
  `state` should be a vector of size `hidden_size` or a matrix of size
  `hidden_size x batch_size`; `cstate` should be a vector of size `memory_size` or a
  matrix of size `memory_size x batch_size`. If not provided, they are assumed to be
  vectors of zeros, initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state.
  `new_state` has size `hidden_size` or `hidden_size x batch_size`; `new_cstate` has
  size `memory_size` or `memory_size x batch_size`.
"""
struct ResLSTMCell{I, H, P, O, R, Q, V, W, G, B, A} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    weight_ph::P
    weight_poh::O
    weight_proj::R
    weight_res::Q
    bias_ih::V
    bias_hh::W
    bias_ph::G
    bias_poh::B
    integration_fn::A
end

@layer ResLSTMCell

function ResLSTMCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        memory_size::Union{Nothing, Int}=nothing,
        init_kernel=glorot_uniform,
        init_recurrent_kernel=glorot_uniform,
        init_peephole_kernel=glorot_uniform,
        init_projection_kernel=glorot_uniform,
        init_residual_kernel=glorot_uniform,
        bias::Bool=true,
        recurrent_bias::Bool=true,
        peephole_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    memory_size = isnothing(memory_size) ? hidden_size : memory_size
    if independent_recurrence && memory_size != hidden_size
        throw(ArgumentError(
            "independent_recurrence requires memory_size == hidden_size; got " *
            "memory_size=$memory_size and hidden_size=$hidden_size"
        ))
    end
    gate_size = 3 * memory_size + hidden_size
    weight_ih = init_kernel(gate_size, input_size)
    weight_hh = if independent_recurrence
        vec(init_recurrent_kernel(gate_size))
    else
        init_recurrent_kernel(gate_size, hidden_size)
    end
    weight_ph = vec(init_peephole_kernel(memory_size * 2))
    weight_poh = init_peephole_kernel(hidden_size, memory_size)
    weight_proj = init_projection_kernel(hidden_size, memory_size)
    weight_res = init_residual_kernel(hidden_size, input_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    bias_ph = create_bias(weight_ph, peephole_bias, size(weight_ph, 1))
    bias_poh = create_bias(weight_poh, peephole_bias, size(weight_poh, 1))
    integration_fn = _integration_fn(integration_mode)
    return ResLSTMCell(weight_ih, weight_hh, weight_ph, weight_poh, weight_proj,
        weight_res, bias_ih, bias_hh, bias_ph, bias_poh, integration_fn)
end

function (lstm::ResLSTMCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(lstm, inp, 1 => size(lstm.weight_ih, 2))
    proj_ih = dense_proj(lstm.weight_ih, inp, lstm.bias_ih)
    proj_hh = dense_proj(lstm.weight_hh, state, lstm.bias_hh)
    proj_ph = dense_proj(lstm.weight_ph, c_state, lstm.bias_ph)
    gates = lstm.integration_fn(proj_ih, proj_hh)
    memory_size = size(lstm.weight_proj, 2)
    if gates isa AbstractVector
        input = gates[1:memory_size]
        forget = gates[(memory_size + 1):(2 * memory_size)]
        cell = gates[(2 * memory_size + 1):(3 * memory_size)]
        output = gates[(3 * memory_size + 1):end]
    else
        input = gates[1:memory_size, :]
        forget = gates[(memory_size + 1):(2 * memory_size), :]
        cell = gates[(2 * memory_size + 1):(3 * memory_size), :]
        output = gates[(3 * memory_size + 1):end, :]
    end
    peeps = chunk(proj_ph, 2; dims=1)
    new_cstate = @. sigmoid_fast(forget + peeps[2]) * c_state +
                    sigmoid_fast(input + peeps[1]) * tanh_fast(cell)
    proj_poh = dense_proj(lstm.weight_poh, new_cstate, lstm.bias_poh)
    output_gate = @. sigmoid_fast(output + proj_poh)
    projection = dense_proj(lstm.weight_proj, tanh_fast.(new_cstate), false)
    residual = dense_proj(lstm.weight_res, inp, false)
    new_state = @. output_gate * (projection + residual)
    return new_state, (new_state, new_cstate)
end

function initialstates(lstm::ResLSTMCell)
    state = zeros_like(lstm.weight_proj, size(lstm.weight_proj, 1))
    second_state = zeros_like(lstm.weight_proj, size(lstm.weight_proj, 2))
    return state, second_state
end

function Base.show(io::IO, lstm::ResLSTMCell)
    print(io, "ResLSTMCell(", size(lstm.weight_ih, 2),
        " => ", size(lstm.weight_proj, 1))
    if size(lstm.weight_proj, 2) != size(lstm.weight_proj, 1)
        print(io, ", memory_size=", size(lstm.weight_proj, 2))
    end
    print(io, ")")
end

@doc raw"""
    ResLSTM(input_size => hidden_size;
        return_state=false,
        kwargs...)

Residual long short-term memory network [Kim2017](@cite).
See [`ResLSTMCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and output dimension of the layer.

# Keyword arguments

- `memory_size`: internal memory cell dimension. Default is `hidden_size`.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_peephole_kernel`: initializer for the peephole weights.
    Default is `glorot_uniform`.
- `init_projection_kernel`: initializer for the cell output projection weights.
    Default is `glorot_uniform`.
- `init_residual_kernel`: initializer for the residual projection weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `peephole_bias`: include peephole to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `return_state`: Option to return the last state together with the output.
  Default is `false`.

# Equations

```math
\begin{aligned}
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{hh} \mathbf{h}(t-1) + \mathbf{p}^{i} \odot
        \mathbf{c}(t-1) + \mathbf{b}^{i} \right), \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{p}^{f} \odot
        \mathbf{c}(t-1) + \mathbf{b}^{f} \right), \\
    \mathbf{g}(t) &= \tanh\left( \mathbf{W}^{g}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{g}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{g} \right), \\
    \mathbf{c}(t) &= \mathbf{f}(t) \odot \mathbf{c}(t-1) +
        \mathbf{i}(t) \odot \mathbf{g}(t), \\
    \mathbf{o}(t) &= \sigma\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{hh} \mathbf{h}(t-1) + \mathbf{W}^{o}_{ch}
        \mathbf{c}(t) + \mathbf{b}^{o} \right), \\
    \mathbf{m}(t) &= \mathbf{W}_{p} \tanh\left( \mathbf{c}(t) \right), \\
    \mathbf{h}(t) &= \mathbf{o}(t) \odot
        \left( \mathbf{m}(t) + \mathbf{W}_{r} \mathbf{x}(t) \right)
\end{aligned}
```

# Forward

    reslstm(inp, (state, cstate))
    reslstm(inp)

## Arguments
- `inp`: The input to the reslstm. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the ResLSTM.
  `state` should be a vector of size `hidden_size` or a matrix of size
  `hidden_size x batch_size`; `cstate` should be a vector of size `memory_size` or a
  matrix of size `memory_size x batch_size`. If not provided, they are assumed to be
  vectors of zeros, initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct ResLSTM{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand ResLSTM

function ResLSTM((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = ResLSTMCell(input_size => hidden_size; kwargs...)
    return ResLSTM{return_state, typeof(cell)}(cell)
end

function functor(rnn::ResLSTM{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> ResLSTM{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, reslstm::ResLSTM)
    print(io, "ResLSTM(", size(reslstm.cell.weight_ih, 2),
        " => ", size(reslstm.cell.weight_proj, 1))
    if size(reslstm.cell.weight_proj, 2) != size(reslstm.cell.weight_proj, 1)
        print(io, ", memory_size=", size(reslstm.cell.weight_proj, 2))
    end
    print(io, ")")
end
