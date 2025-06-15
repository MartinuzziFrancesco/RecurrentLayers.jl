#https://arxiv.org/abs/1602.02218
@doc raw"""
    TRNNCell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Strongly typed recurrent unit](https://arxiv.org/abs/1602.02218).
See [`TRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: activation function. Default is `tanh`.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \mathbf{W}_{ih} \mathbf{x}(t) + \mathbf{b}^{z} \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}_{fh} \mathbf{x}(t) +
        \mathbf{b}^{f} \right) \\
    \mathbf{h}(t) &= \mathbf{f}(t) \odot \mathbf{h}(t-1) + \left(1 -
        \mathbf{f}(t)\right) \odot \mathbf{z}(t)
\end{aligned}
```

# Forward

    trnncell(inp, state)
    trnncell(inp)

## Arguments
- `inp`: The input to the trnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the TRNNCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct TRNNCell{I, V, A} <: AbstractRecurrentCell
    Wi::I
    bias::V
    activation::A
end

@layer TRNNCell

function TRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int}, activation=tanh_fast;
        init_kernel=glorot_uniform, bias::Bool=true)
    Wi = init_kernel(hidden_size * 2, input_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    return TRNNCell(Wi, b, activation)
end

function (trnn::TRNNCell)(inp::AbstractVecOrMat, state)
    _size_check(trnn, inp, 1 => size(trnn.Wi, 2))
    Wi, b, activation = trnn.Wi, trnn.bias, trnn.activation
    #split
    gxs = chunk(Wi * inp .+ b, 2; dims=1)

    forget_gate = activation.(gxs[2])
    t_ones = eltype(Wi)(1.0f0)
    new_state = @. forget_gate * state + (t_ones - forget_gate) * gxs[1]
    return new_state, new_state
end

function initialstates(trnn::TRNNCell)
    return zeros_like(trnn.Wi, size(trnn.Wi, 1) ÷ 2)
end

function Base.show(io::IO, trnn::TRNNCell)
    print(io, "TRNNCell(", size(trnn.Wi, 2), " => ", size(trnn.Wi, 1) ÷ 2, ")")
end

@doc raw"""
    TRNN(input_size => hidden_size, [activation];
        return_state = false, kwargs...)

[Strongly typed recurrent unit](https://arxiv.org/abs/1602.02218).
See [`TRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: activation function. Default is `tanh`.

# Keyword arguments

- `return_state`: Option to return the last state together with the output.
  Default is `false`.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \mathbf{W}_{ih} \mathbf{x}(t) + \mathbf{b}^{z} \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}_{fh} \mathbf{x}(t) +
        \mathbf{b}^{f} \right) \\
    \mathbf{h}(t) &= \mathbf{f}(t) \odot \mathbf{h}(t-1) + \left(1 -
        \mathbf{f}(t)\right) \odot \mathbf{z}(t)
\end{aligned}
```

# Forward

    trnn(inp, state)
    trnn(inp)

## Arguments
- `inp`: The input to the trnn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the TRNN. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct TRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand TRNN

function TRNN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = TRNNCell(input_size => hidden_size; kwargs...)
    return TRNN{return_state, typeof(cell)}(cell)
end

function functor(rnn::TRNN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> TRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, trnn::TRNN)
    print(io, "TRNN(", size(trnn.cell.Wi, 2), " => ", size(trnn.cell.Wi, 1) ÷ 2)
    print(io, ")")
end

@doc raw"""
    TGRUCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Strongly typed gated recurrent unit](https://arxiv.org/abs/1602.02218).
See [`TGRU`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{x}(t-1) + \mathbf{b}^{z} \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{x}(t-1) + \mathbf{b}^{f} \right) \\
    \mathbf{o}(t) &= \tau\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{hh} \mathbf{x}(t-1) + \mathbf{b}^{o} \right) \\
    \mathbf{h}(t) &= \mathbf{f}(t) \odot \mathbf{h}(t-1) +
        \mathbf{z}(t) \odot \mathbf{o}(t)
\end{aligned}
```

# Forward

    tgrucell(inp, state)
    tgrucell(inp)

## Arguments
- `inp`: The input to the tgrucell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the TGRUCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, inp)` is the new hidden state together with the current input. 
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct TGRUCell{I, H, V} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

@layer TGRUCell

function TGRUCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size * 3, input_size)
    Wh = init_recurrent_kernel(hidden_size * 3, input_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    return TGRUCell(Wi, Wh, b)
end

function (tgru::TGRUCell)(inp::AbstractVecOrMat, (state, prev_inp))
    #checks and variables
    _size_check(tgru, inp, 1 => size(tgru.Wi, 2))
    Wi, Wh, b = tgru.Wi, tgru.Wh, tgru.bias
    #split
    gxs = chunk(Wi * inp .+ b, 3; dims=1)
    ghs = chunk(Wh * prev_inp, 3; dims=1)
    #equations
    reset_gate = @. gxs[1] + ghs[1]
    update_gate = @. sigmoid_fast(gxs[2] + ghs[2])
    candidate_state = @. tanh_fast(gxs[3] + ghs[3])
    new_state = @. update_gate * state + reset_gate * candidate_state
    return new_state, (new_state, inp)
end

function initialstates(tgru::TGRUCell)
    initial_state = zeros_like(tgru.Wi, size(tgru.Wi, 1) ÷ 3)
    initial_inp = zeros_like(tgru.Wi, size(tgru.Wi, 2))
    return initial_state, initial_inp
end

function Base.show(io::IO, tgru::TGRUCell)
    print(io, "TGRUCell(", size(tgru.Wi, 2), " => ", size(tgru.Wi, 1) ÷ 3, ")")
end

@doc raw"""
    TGRU(input_size => hidden_size;
        return_state = false, kwargs...)

[Strongly typed recurrent gated unit](https://arxiv.org/abs/1602.02218).
See [`TGRUCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `return_state`: Option to return the last state together with the output.
  Default is `false`.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{x}(t-1) + \mathbf{b}^{z} \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{x}(t-1) + \mathbf{b}^{f} \right) \\
    \mathbf{o}(t) &= \tau\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{hh} \mathbf{x}(t-1) + \mathbf{b}^{o} \right) \\
    \mathbf{h}(t) &= \mathbf{f}(t) \odot \mathbf{h}(t-1) +
        \mathbf{z}(t) \odot \mathbf{o}(t)
\end{aligned}
```

# Forward

    tgru(inp, state)
    tgru(inp)

## Arguments
- `inp`: The input to the tgru. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the TGRU. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct TGRU{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand TGRU

function TGRU((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = TGRUCell(input_size => hidden_size; kwargs...)
    return TGRU{return_state, typeof(cell)}(cell)
end

function functor(tgru::TGRU{S}) where {S}
    params = (cell=tgru.cell,)
    reconstruct = p -> TGRU{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, tgru::TGRU)
    print(io, "TGRU(", size(tgru.cell.Wi, 2), " => ", size(tgru.cell.Wi, 1) ÷ 3)
    print(io, ")")
end

@doc raw"""
    TLSTMCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Strongly typed long short term memory cell](https://arxiv.org/abs/1602.02218).
See [`TLSTM`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{x}(t-1) + \mathbf{b}^{z} \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{x}(t-1) + \mathbf{b}^{f} \right) \\
    \mathbf{o}(t) &= \tau\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{hh} \mathbf{x}(t-1) + \mathbf{b}^{o} \right) \\
    \mathbf{c}(t) &= \mathbf{f}(t) \odot \mathbf{c}(t-1) +
        \left(1 - \mathbf{f}(t)\right) \odot \mathbf{z}(t) \\
    \mathbf{h}(t) &= \mathbf{c}(t) \odot \mathbf{o}(t)
\end{aligned}
```

# Forward

    tlstmcell(inp, state)
    tlstmcell(inp)

## Arguments
- `inp`: The input to the tlstmcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the TLSTMCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate, inp)` is the new hidden and cell state, together
  with the current input. 
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct TLSTMCell{I, H, V} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

@layer TLSTMCell

function TLSTMCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size * 3, input_size)
    Wh = init_recurrent_kernel(hidden_size * 3, input_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    return TLSTMCell(Wi, Wh, b)
end

function (tlstm::TLSTMCell)(inp::AbstractVecOrMat, (state, c_state, prev_inp))
    #checks and variables
    _size_check(tlstm, inp, 1 => size(tlstm.Wi, 2))
    Wi, Wh, b = tlstm.Wi, tlstm.Wh, tlstm.bias
    #split
    gxs = chunk(Wi * inp .+ b, 3; dims=1)
    ghs = chunk(Wh * prev_inp, 3; dims=1)
    #equations
    one_vec = eltype(Wi)(1.0f0)
    reset_gate = @. gxs[1] + ghs[1]
    update_gate = @. sigmoid_fast(gxs[2] + ghs[2])
    candidate_state = @. tanh_fast(gxs[3] + ghs[3])
    new_cstate = @. update_gate * c_state + (one_vec - update_gate) * reset_gate
    new_state = @. new_cstate * candidate_state
    return new_state, (new_state, new_cstate, inp)
end

function initialstates(tlstm::TLSTMCell)
    initial_state = zeros_like(tlstm.Wi, size(tlstm.Wi, 1) ÷ 3)
    initial_cstate = zeros_like(tlstm.Wi, size(tlstm.Wi, 1) ÷ 3)
    initial_inp = zeros_like(tlstm.Wi, size(tlstm.Wi, 2))
    return initial_state, initial_cstate, initial_inp
end

function Base.show(io::IO, tlstm::TLSTMCell)
    print(io, "TLSTMCell(", size(tlstm.Wi, 2), " => ", size(tlstm.Wi, 1) ÷ 3, ")")
end

@doc raw"""
    TLSTM(input_size => hidden_size;
        return_state = false, kwargs...)

[Strongly typed long short term memory](https://arxiv.org/abs/1602.02218).
See [`TLSTMCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `return_state`: Option to return the last state together with the output.
  Default is `false`.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{x}(t-1) + \mathbf{b}^{z} \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{x}(t-1) + \mathbf{b}^{f} \right) \\
    \mathbf{o}(t) &= \tau\left( \mathbf{W}^{o}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{o}_{hh} \mathbf{x}(t-1) + \mathbf{b}^{o} \right) \\
    \mathbf{c}(t) &= \mathbf{f}(t) \odot \mathbf{c}(t-1) +
        \left(1 - \mathbf{f}(t)\right) \odot \mathbf{z}(t) \\
    \mathbf{h}(t) &= \mathbf{c}(t) \odot \mathbf{o}(t)
\end{aligned}
```

# Forward

    tlstm(inp, state)
    tlstm(inp)

## Arguments
- `inp`: The input to the tlstm. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the TLSTM. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden states `new_states` and
  the last state of the iteration.
"""
struct TLSTM{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand TLSTM

function TLSTM((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = TLSTMCell(input_size => hidden_size; kwargs...)
    return TLSTM{return_state, typeof(cell)}(cell)
end

function functor(tlstm::TLSTM{S}) where {S}
    params = (cell=tlstm.cell,)
    reconstruct = p -> TLSTM{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, tlstm::TLSTM)
    print(io, "TLSTM(", size(tlstm.cell.Wi, 2), " => ", size(tlstm.cell.Wi, 1) ÷ 3)
    print(io, ")")
end
