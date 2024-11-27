#https://arxiv.org/pdf/1803.10225
struct LiGRUCell{I, H, V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer LiGRUCell

@doc raw"""
    LiGRUCell((input_size => hidden_size)::Pair;
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

    rnncell(inp, [state])
"""
function LiGRUCell((input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true)

    Wi = init_kernel(hidden_size * 2, input_size)
    Wh = init_recurrent_kernel(hidden_size * 2, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))

    return LiGRUCell(Wi, Wh, b)
end

function (ligru::LiGRUCell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(ligru.Wh, 2))
    return ligru(inp, state)
end

function (ligru::LiGRUCell)(inp::AbstractVecOrMat, state)
    _size_check(ligru, inp, 1 => size(ligru.Wi,2))
    Wi, Wh, b = ligru.Wi, ligru.Wh, ligru.bias
    #split
    gxs = chunk(Wi * inp, 2, dims=1)
    ghs = chunk(Wh * state .+ b, 2, dims=1)
    #compute
    forget_gate = @. sigmoid_fast(gxs[1] + ghs[1])
    candidate_hidden = @. tanh_fast(gxs[2] + ghs[2])
    new_state = forget_gate .* state .+ (1 .- forget_gate) .* candidate_hidden
    return new_state
end


struct LiGRU{M}
    cell::M
end
  
Flux.@layer :expand LiGRU

@doc raw"""
    LiGRU((input_size => hidden_size)::Pair; kwargs...)

[Light gated recurrent network](https://arxiv.org/pdf/1803.10225).
The implementation does not include the batch normalization as
described in the original paper.
See [`LiGRUCell`](@ref) for a layer that processes a single sequence.

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
"""
function LiGRU((input_size, hidden_size)::Pair; kwargs...)
    cell = LiGRUCell(input_size => hidden_size; kwargs...)
    return LiGRU(cell)
end
  
function (ligru::LiGRU)(inp)
    state = zeros_like(inp, size(ligru.cell.Wh, 2))
    return ligru(inp, state)
end
  
function (ligru::LiGRU)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = ligru.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end


Base.show(io::IO, ligru::LiGRUCell) =
    print(io, "LiGRUCell(", size(ligru.Wi, 2), " => ", size(ligru.Wi, 1) ÷ 2, ")")
