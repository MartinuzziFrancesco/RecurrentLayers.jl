#https://arxiv.org/abs/1711.06788
@doc raw"""
    MinimalRNNCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, encoder_bias = true)

Minimal recurrent neural network unit [^Zhang2017].
See [`MinimalRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.
- `encoder_bias`: include a bias in the encoder or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \Phi(\mathbf{x}(t)) = \tanh\left( \mathbf{W}_{xz}
        \mathbf{x}(t) + \mathbf{b}^{z} \right), \\
    \mathbf{u}(t) &= \sigma\left( \mathbf{W}_{hh}^{u} \mathbf{h}(t-1) +
        \mathbf{W}_{zh}^{u} \mathbf{z}(t) + \mathbf{b}^{u} \right), \\
    \mathbf{h}(t) &= \mathbf{u}(t) \circ \mathbf{h}(t-1) + \left(1 -
        \mathbf{u}(t)\right) \circ \mathbf{z}(t)
\end{aligned}
```

# Forward

    minimalrnncell(inp, state)
    minimalrnncell(inp)

## Arguments
- `inp`: The input to the minimalrnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the MinimalRNNCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state. 
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.

[^Zhang2017]: Zhang, M. et al.  
    _Minimal RNN: Toward more interpretable and trainable recurrent nets._  
    NeurIPS 2017.
"""
struct MinimalRNNCell{I, H, Z, V, E} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    Wz::Z
    bias::V
    encoder_bias::E
end

@layer MinimalRNNCell

function MinimalRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, encoder_bias::Bool=true)
    Wi = init_kernel(hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size, hidden_size)
    Wz = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    eb = create_bias(Wi, encoder_bias, size(Wi, 1))
    return MinimalRNNCell(Wi, Wh, Wz, b, eb)
end

function (minimal::MinimalRNNCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(minimal, inp, 1 => size(minimal.Wi, 2))
    Wi, Wh, Wz = minimal.Wi, minimal.Wh, minimal.Wz
    b, eb = minimal.bias, minimal.encoder_bias
    #compute
    new_cstate = tanh_fast.(Wi * inp .+ eb)
    update_gate = sigmoid_fast.(Wh * state .+ Wz * c_state .+ b)
    new_state = update_gate .* state .+ (eltype(Wi)(1.0) .- update_gate) .* new_cstate
    return new_state, (new_state, new_cstate)
end

function Base.show(io::IO, minimal::MinimalRNNCell)
    print(io, "MinimalRNNCell(", size(minimal.Wi, 2), " => ", size(minimal.Wi, 1), ")")
end

@doc raw"""
    MinimalRNN(input_size => hidden_size;
        return_state = false, kwargs...)

Minimal recurrent neural network [^Zhang2017].
See [`MinimalRNNCell`](@ref) for a layer that processes a single sequence.

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
- `encoder_bias`: include a bias in the encoder or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \Phi(\mathbf{x}(t)) = \tanh\left( \mathbf{W}_{xz}
        \mathbf{x}(t) + \mathbf{b}^{z} \right), \\
    \mathbf{u}(t) &= \sigma\left( \mathbf{W}_{hh}^{u} \mathbf{h}(t-1) +
        \mathbf{W}_{zh}^{u} \mathbf{z}(t) + \mathbf{b}^{u} \right), \\
    \mathbf{h}(t) &= \mathbf{u}(t) \circ \mathbf{h}(t-1) + \left(1 -
        \mathbf{u}(t)\right) \circ \mathbf{z}(t)
\end{aligned}
```

# Forward

    minimalrnn(inp, (state, c_state))
    minimalrnn(inp)

## Arguments
- `inp`: The input to the `minimalrnn`. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the `MinimalRNN`. 
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.

[^Zhang2017]: Zhang, M. et al.  
    _Minimal RNN: Toward more interpretable and trainable recurrent nets._  
    NeurIPS 2017.
"""
struct MinimalRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand MinimalRNN

function MinimalRNN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = MinimalRNNCell(input_size => hidden_size; kwargs...)
    return MinimalRNN{return_state, typeof(cell)}(cell)
end

function functor(rnn::MinimalRNN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> MinimalRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, minimal::MinimalRNN)
    print(io, "MinimalRNN(", size(minimal.cell.Wi, 2), " => ", size(minimal.cell.Wi, 1))
    print(io, ")")
end
