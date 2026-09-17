#https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
@doc raw"""
    PeepholeLSTMCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        init_peephole_kernel = glorot_uniform,
        bias = true, recurrent_bias = true, peephole_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Peephole long short term memory cell [Gers2002](@cite).
See [`PeepholeLSTM`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_peephole_kernel`: initializer for the hidden to peephole weights.
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
    \mathbf{z}(t) &= \tanh\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{z} \right), \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{hh} \mathbf{h}(t-1) + \mathbf{w}^{i}_{ph} \odot
        \mathbf{c}(t-1) + \mathbf{b}^{i} \right), \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{w}^{f}_{ph} \odot
        \mathbf{c}(t-1) + \mathbf{b}^{f} \right), \\
    \mathbf{c}(t) &= \mathbf{f}(t) \odot \mathbf{c}(t-1) + \mathbf{i}(t)
        \odot \mathbf{z}(t), \\
    \mathbf{o}(t) &= \sigma\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{hh} \mathbf{h}(t-1) + \mathbf{w}^{o}_{ph} \odot
        \mathbf{c}(t) + \mathbf{b}^{o} \right), \\
    \mathbf{h}(t) &= \mathbf{o}(t) \odot \tanh\left( \mathbf{c}(t) \right)
\end{aligned}
```

# Forward

    peepholelstmcell(inp, (state, cstate))
    peepholelstmcell(inp)

## Arguments

- `inp`: The input to the peepholelstmcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the PeepholeLSTMCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state.
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct PeepholeLSTMCell{I, H, P, V, W, G, A} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    weight_ph::P
    bias_ih::V
    bias_hh::W
    bias_ph::G
    integration_fn::A
end

@layer PeepholeLSTMCell

function PeepholeLSTMCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        init_peephole_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true, peephole_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(hidden_size * 4, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 4)
    weight_ph = vec(init_peephole_kernel(hidden_size * 3))
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    bias_ph = create_bias(weight_ph, peephole_bias, size(weight_ph, 1))
    integration_fn = _integration_fn(integration_mode)
    return PeepholeLSTMCell(weight_ih, weight_hh, weight_ph, bias_ih,
        bias_hh, bias_ph, integration_fn)
end

_split_ph_bias(bias::Bool, hidden_size::Int) = bias, bias
function _split_ph_bias(bias::AbstractVector, hidden_size::Int)
    return bias[1:(2 * hidden_size)], bias[(2 * hidden_size + 1):end]
end

function (lstm::PeepholeLSTMCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(lstm, inp, 1 => size(lstm.weight_ih, 2))
    hidden_size = size(lstm.weight_ih, 1) ÷ 4
    proj_ih = dense_proj(lstm.weight_ih, inp, lstm.bias_ih)
    proj_hh = dense_proj(lstm.weight_hh, state, lstm.bias_hh)
    gates = lstm.integration_fn(proj_ih, proj_hh)
    input, forget, cell, output = chunk(gates, 4; dims=1)

    # input/forget peepholes read c(t-1); the output peephole reads c(t)
    bias_ph_if, bias_ph_o = _split_ph_bias(lstm.bias_ph, hidden_size)
    weight_ph_if = @view lstm.weight_ph[1:(2 * hidden_size)]
    weight_ph_o = @view lstm.weight_ph[(2 * hidden_size + 1):end]
    peep_i, peep_f = chunk(dense_proj(weight_ph_if, c_state, bias_ph_if), 2; dims=1)
    new_cstate = @. sigmoid_fast(forget + peep_f) * c_state +
                    sigmoid_fast(input + peep_i) * tanh_fast(cell)
    peep_o = dense_proj(weight_ph_o, new_cstate, bias_ph_o)
    new_state = @. sigmoid_fast(output + peep_o) * tanh_fast(new_cstate)
    return new_state, (new_state, new_cstate)
end

function initialstates(lstm::PeepholeLSTMCell)
    state = zeros_like(lstm.weight_hh, size(lstm.weight_hh, 1) ÷ 4)
    second_state = zeros_like(lstm.weight_hh, size(lstm.weight_hh, 1) ÷ 4)
    return state, second_state
end

function Base.show(io::IO, lstm::PeepholeLSTMCell)
    print(io, "PeepholeLSTMCell(", size(lstm.weight_ih, 2),
        " => ", size(lstm.weight_ih, 1) ÷ 4, ")")
end

@doc raw"""
    PeepholeLSTM(input_size => hidden_size;
        return_state=false,
        kwargs...)

Peephole long short term memory network [Gers2002](@cite).
See [`PeepholeLSTMCell`](@ref) for a layer that processes a single sequence.

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
    \mathbf{z}(t) &= \tanh\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{z} \right), \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{hh} \mathbf{h}(t-1) + \mathbf{w}^{i}_{ph} \odot
        \mathbf{c}(t-1) + \mathbf{b}^{i} \right), \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{w}^{f}_{ph} \odot
        \mathbf{c}(t-1) + \mathbf{b}^{f} \right), \\
    \mathbf{c}(t) &= \mathbf{f}(t) \odot \mathbf{c}(t-1) + \mathbf{i}(t)
        \odot \mathbf{z}(t), \\
    \mathbf{o}(t) &= \sigma\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{hh} \mathbf{h}(t-1) + \mathbf{w}^{o}_{ph} \odot
        \mathbf{c}(t) + \mathbf{b}^{o} \right), \\
    \mathbf{h}(t) &= \mathbf{o}(t) \odot \tanh\left( \mathbf{c}(t) \right)
\end{aligned}
```

# Forward

    peepholelstm(inp, (state, cstate))
    peepholelstm(inp)

## Arguments
- `inp`: The input to the peepholelstm. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the PeepholeLSTM.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct PeepholeLSTM{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand PeepholeLSTM

function PeepholeLSTM((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = PeepholeLSTMCell(input_size => hidden_size; kwargs...)
    return PeepholeLSTM{return_state, typeof(cell)}(cell)
end

function functor(rnn::PeepholeLSTM{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> PeepholeLSTM{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, peepholelstm::PeepholeLSTM)
    print(io, "PeepholeLSTM(", size(peepholelstm.cell.weight_ih, 2),
        " => ", size(peepholelstm.cell.weight_ih, 1) ÷ 4)
    print(io, ")")
end
