#https://www.mdpi.com/2079-9292/13/16/3204
struct LightRUCell{I,H,V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer LightRUCell

@doc raw"""
    LightRUCell((input_size => hidden_size)::Pair;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Light recurrent unit](https://www.mdpi.com/2079-9292/13/16/3204).
See [`LightRU`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
\tilde{h}_t &= \tanh(W_h x_t), \\
f_t         &= \delta(W_f x_t + U_f h_{t-1} + b_f), \\
h_t         &= (1 - f_t) \odot h_{t-1} + f_t \odot \tilde{h}_t.
\end{aligned}
```

# Forward

    rnncell(inp, [state])
"""
function LightRUCell((input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true)
    Wi = init_kernel(2 * hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wh, 1))

    return LightRUCell(Wi, Wh, b)
end

function (lru::LightRUCell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(lru.Wh, 2))
    return lru(inp, state)
end

function (lru::LightRUCell)(inp::AbstractVecOrMat, state)
    _size_check(lru, inp, 1 => size(lru.Wi,2))
    Wi, Wh, b = lru.Wi, lru.Wh, lru.bias

    #split
    gxs = chunk(Wi * inp, 2, dims=1)

    #compute
    candidate_state = @. tanh_fast(gxs[1])
    forget_gate = sigmoid_fast(gxs[2] .+ Wh * state .+ b)
    new_state = @. (1 - forget_gate) * state + forget_gate * candidate_state
    return new_state
end

Base.show(io::IO, lru::LightRUCell) =
    print(io, "LightRUCell(", size(lru.Wi, 2), " => ", size(lru.Wi, 1)÷2, ")")



struct LightRU{M}
    cell::M
end
  
Flux.@layer :expand LightRU

@doc raw"""
    LightRU((input_size => hidden_size)::Pair; kwargs...)

[Light recurrent unit network](https://www.mdpi.com/2079-9292/13/16/3204).
See [`LightRUCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
\tilde{h}_t &= \tanh(W_h x_t), \\
f_t         &= \delta(W_f x_t + U_f h_{t-1} + b_f), \\
h_t         &= (1 - f_t) \odot h_{t-1} + f_t \odot \tilde{h}_t.
\end{aligned}
```
"""
function LightRU((input_size, hidden_size)::Pair; kwargs...)
    cell = LightRUCell(input_size => hidden_size; kwargs...)
    return LightRU(cell)
end
  
function (lru::LightRU)(inp)
    state = zeros_like(inp, size(lru.cell.Wh, 2))
    return lru(inp, state)
end
  
function (lru::LightRU)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = lru.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end
