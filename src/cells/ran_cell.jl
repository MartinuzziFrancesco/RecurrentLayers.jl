#https://arxiv.org/pdf/1705.07393
@doc raw"""
    RANCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Recurrent Additive Network cell](https://arxiv.org/pdf/1705.07393).
See [`RAN`](@ref) for a layer that processes entire sequences.

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
    \tilde{\mathbf{c}}(t) &= \mathbf{W}^{c}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{c}, \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{i} \right), \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{f} \right), \\
    \mathbf{c}(t) &= \mathbf{i}(t) \odot \tilde{\mathbf{c}}(t) +
        \mathbf{f}(t) \odot \mathbf{c}(t-1), \\
    \mathbf{h}(t) &= g\left( \mathbf{c}(t) \right)
\end{aligned}
```

# Forward

    rancell(inp, (state, cstate))
    rancell(inp)

## Arguments
- `inp`: The input to the rancell. It should be a vector of size `input_size`
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
struct RANCell{I, H, V} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

@layer RANCell

function RANCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(3 * hidden_size, input_size)
    Wh = init_recurrent_kernel(2 * hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wh, 1))
    return RANCell(Wi, Wh, b)
end

function (ran::RANCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(ran, inp, 1 => size(ran.Wi, 2))
    Wi, Wh, b = ran.Wi, ran.Wh, ran.bias
    #split
    gxs = chunk(Wi * inp, 3; dims=1)
    ghs = chunk(Wh * state .+ b, 2; dims=1)
    #compute
    input_gate = @. sigmoid_fast(gxs[2] + ghs[1])
    forget_gate = @. sigmoid_fast(gxs[3] + ghs[2])
    candidate_state = @. input_gate * gxs[1] + forget_gate * c_state
    new_state = @. tanh_fast(candidate_state)
    return new_state, (new_state, candidate_state)
end

function Base.show(io::IO, ran::RANCell)
    print(io, "RANCell(", size(ran.Wi, 2), " => ", size(ran.Wi, 1) ÷ 3, ")")
end

@doc raw"""
    RAN(input_size => hidden_size;
        return_state = false, kwargs...)

[Recurrent Additive Network cell](https://arxiv.org/pdf/1705.07393).
See [`RANCell`](@ref) for a layer that processes a single sequence.

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
    \tilde{\mathbf{c}}(t) &= \mathbf{W}^{c}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{c} \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{i} \right) \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{f} \right) \\
    \mathbf{c}(t) &= \mathbf{i}(t) \odot \tilde{\mathbf{c}}(t) +
        \mathbf{f}(t) \odot \mathbf{c}(t-1) \\
    \mathbf{h}(t) &= g\left( \mathbf{c}(t) \right)
\end{aligned}
```

# Forward

    ran(inp, (state, cstate))
    ran(inp)

## Arguments
- `inp`: The input to the ran. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the RAN. 
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct RAN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand RAN

function RAN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = RANCell(input_size => hidden_size; kwargs...)
    return RAN{return_state, typeof(cell)}(cell)
end

function functor(rnn::RAN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> RAN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, ran::RAN)
    print(io, "RAN(", size(ran.cell.Wi, 2), " => ", size(ran.cell.Wi, 1))
    print(io, ")")
end
