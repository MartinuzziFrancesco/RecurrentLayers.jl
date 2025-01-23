#https://arxiv.org/pdf/1803.10225
@doc raw"""
    LiGRUCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Light gated recurrent unit](https://arxiv.org/pdf/1803.10225).
The implementation does not include the batch normalization as
described in the original paper.
See [`LiGRU`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1}), \\
\tilde{h}_t &= \text{ReLU}(W_h x_t + U_h h_{t-1}), \\
h_t &= z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t
\end{aligned}
```

# Forward

    ligrucell(inp, state)
    ligrucell(inp)

## Arguments
- `inp`: The input to the ligrucell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the LiGRUCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state `new_state`, 
  a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct LiGRUCell{I, H, V} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

@layer LiGRUCell

function LiGRUCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size * 2, input_size)
    Wh = init_recurrent_kernel(hidden_size * 2, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))

    return LiGRUCell(Wi, Wh, b)
end

function (ligru::LiGRUCell)(inp::AbstractVecOrMat, state)
    _size_check(ligru, inp, 1 => size(ligru.Wi, 2))
    Wi, Wh, b = ligru.Wi, ligru.Wh, ligru.bias
    #split
    gxs = chunk(Wi * inp, 2; dims=1)
    ghs = chunk(Wh * state .+ b, 2; dims=1)
    #compute
    forget_gate = @. sigmoid_fast(gxs[1] + ghs[1])
    candidate_hidden = @. tanh_fast(gxs[2] + ghs[2])
    new_state = forget_gate .* state .+ (1 .- forget_gate) .* candidate_hidden
    return new_state, new_state
end

function Base.show(io::IO, ligru::LiGRUCell)
    print(io, "LiGRUCell(", size(ligru.Wi, 2), " => ", size(ligru.Wi, 1) ÷ 2, ")")
end

@doc raw"""
    LiGRU(input_size => hidden_size;
        return_state = false, kwargs...)

[Light gated recurrent network](https://arxiv.org/pdf/1803.10225).
The implementation does not include the batch normalization as
described in the original paper.
See [`LiGRUCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `return_state`: Option to return the last state together with the output. Default is `false`.
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1}), \\
\tilde{h}_t &= \text{ReLU}(W_h x_t + U_h h_{t-1}), \\
h_t &= z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t
\end{aligned}
```

# Forward

    ligru(inp, state)
    ligru(inp)

## Arguments
- `inp`: The input to the ligru. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the LiGRU. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct LiGRU{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand LiGRU

function LiGRU((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = LiGRUCell(input_size => hidden_size; kwargs...)
    return LiGRU{return_state, typeof(cell)}(cell)
end

function functor(rnn::LiGRU{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> LiGRU{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, ligru::LiGRU)
    print(io, "LiGRU(", size(ligru.cell.Wi, 2), " => ", size(ligru.cell.Wi, 1))
    print(io, ")")
end
