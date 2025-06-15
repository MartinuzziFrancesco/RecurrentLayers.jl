#https://arxiv.org/abs/1804.04849
@doc raw"""
    JANETCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, beta_value=1.0)

[Just another network unit](https://arxiv.org/abs/1804.04849).
See [`JANET`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.
- `beta_value`: control over the input data flow.
  Default is 1.0.

# Equations

```math
\begin{aligned}
    \mathbf{s}(t) &= \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{W}^{f}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{f}, \\
    \tilde{\mathbf{c}}(t) &= \tanh\left( \mathbf{W}^{c}_{hh} \mathbf{h}(t-1) +
        \mathbf{W}^{c}_{ih} \mathbf{x}(t) + \mathbf{b}^{c} \right), \\
    \mathbf{c}(t) &= \sigma\left( \mathbf{s}(t) \right) \odot \mathbf{c}(t-1) +
        \left( 1 - \sigma\left( \mathbf{s}(t) - \beta \right) \right) \odot
        \tilde{\mathbf{c}}(t), \\
    \mathbf{h}(t) &= \mathbf{c}(t).
\end{aligned}
```

# Forward

    janetcell(inp, (state, cstate))
    janetcell(inp)

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
struct JANETCell{I, H, B, V} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    beta::B
    bias::V
end

@layer JANETCell trainable=(Wi, Wh, bias)

function JANETCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, beta_value::AbstractFloat=1.0f0)
    Wi = init_kernel(hidden_size * 2, input_size)
    Wh = init_recurrent_kernel(hidden_size * 2, hidden_size)
    beta = fill(eltype(Wi)(beta_value), hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))

    return JANETCell(Wi, Wh, beta, b)
end

function (janet::JANETCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(janet, inp, 1 => size(janet.Wi, 2))
    Wi, Wh, b, beta = janet.Wi, janet.Wh, janet.bias, janet.beta
    #split
    gxs = chunk(Wi * inp .+ b, 2; dims=1)
    ghs = chunk(Wh * state, 2; dims=1)

    linear_gate = gxs[1] .+ ghs[1]
    candidate_state = @. tanh_fast(gxs[2] + ghs[2])
    t_ones = eltype(candidate_state)(1.0f0)
    new_cstate = @. sigmoid_fast(linear_gate) * c_state +
                    (t_ones -
                     sigmoid_fast(linear_gate - beta)) *
                    candidate_state
    new_state = new_cstate

    return new_state, (new_state, new_cstate)
end

function Base.show(io::IO, janet::JANETCell)
    print(io, "JANETCell(", size(janet.Wi, 2), " => ", size(janet.Wi, 1) ÷ 2, ")")
end

@doc raw"""
    JANET(input_size => hidden_size;
        return_state = false, kwargs...)

[Just another network](https://arxiv.org/abs/1804.04849).
See [`JANETCell`](@ref) for a layer that processes a single sequence.

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
- `beta_value`: control over the input data flow.
  Default is 1.0.

# Equations

```math
\begin{aligned}
    \mathbf{s}(t) &= \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{W}^{f}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{f}, \\
    \tilde{\mathbf{c}}(t) &= \tanh\left( \mathbf{W}^{c}_{hh} \mathbf{h}(t-1) +
        \mathbf{W}^{c}_{ih} \mathbf{x}(t) + \mathbf{b}^{c} \right), \\
    \mathbf{c}(t) &= \sigma\left( \mathbf{s}(t) \right) \odot \mathbf{c}(t-1) +
        \left( 1 - \sigma\left( \mathbf{s}(t) - \beta \right) \right) \odot
        \tilde{\mathbf{c}}(t), \\
    \mathbf{h}(t) &= \mathbf{c}(t).
\end{aligned}
```

# Forward

    janet(inp, (state, cstate))
    janet(inp)

## Arguments
- `inp`: The input to the janet. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the JANET. 
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct JANET{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand JANET

function JANET((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = JANETCell(input_size => hidden_size; kwargs...)
    return JANET{return_state, typeof(cell)}(cell)
end

function functor(janet::JANET{S}) where {S}
    params = (cell=janet.cell,)
    reconstruct = p -> JANET{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, janet::JANET)
    print(io, "JANET(", size(janet.cell.Wi, 2), " => ", size(janet.cell.Wi, 1))
    print(io, ")")
end
