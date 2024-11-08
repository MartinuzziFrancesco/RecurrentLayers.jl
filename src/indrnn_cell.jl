#https://arxiv.org/pdf/1803.04831
struct IndRNNCell{F,I,H,V}
    σ::F
    Wi::I
    u::H
    b::V
end

Flux.@layer IndRNNCell

@doc raw"""
    IndRNNCell((in => out)::Pair, σ=relu;
        kernel_init = glorot_uniform,
        recurrent_kernel_init = glorot_uniform,
        bias = true)

# Arguments

- `in => out`: input and inner dimension of the layer
- `σ`: activation function. Default is `tanh`
- `kernel_init`: initializer for the input to hidden weights
- `recurrent_kernel_init`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\mathbf{h}_{t+1} = \sigma(\mathbf{W} \mathbf{x}_t + \mathbf{u} \odot \mathbf{h}_{t} + \mathbf{b})
```

# Forward

    rnncell(inp, [state])

"""
function IndRNNCell((in, out)::Pair, σ=relu;
    kernel_init = glorot_uniform,
    recurrent_kernel_init = glorot_uniform,
    bias = true)
    Wi = kernel_init(out, in)
    u = recurrent_kernel_init(out)
    b = create_bias(Wi, bias, size(Wi, 1))
    return IndRNNCell(σ, Wi, u, b)
end

function (indrnn::IndRNNCell)(x::AbstractVecOrMat)
    state = zeros_like(x, size(indrnn.u, 1))
    return indrnn(x, state)
end

function (indrnn::IndRNNCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(indrnn, inp, 1 => size(indrnn.Wi, 2))
    σ = NNlib.fast_act(indrnn.σ, inp)
    state = σ.(indrnn.Wi*inp .+ indrnn.u.*state .+ indrnn.b)
    return state
end

function Base.show(io::IO, indrnn::IndRNNCell)
    print(io, "IndRNNCell(", size(indrnn.Wi, 2), " => ", size(indrnn.Wi, 1))
    print(io, ", ", indrnn.σ)
    print(io, ")")
end

struct IndRNN{M}
    cell::M
end
  
Flux.@layer :expand IndRNN

"""
    IndRNN((in, out)::Pair, σ = tanh; kwargs...)
"""
function IndRNN((in, out)::Pair, σ = tanh; kwargs...)
    cell = IndRNNCell(in => out, σ; kwargs...)
    return IndRNN(cell)
end
  
function (indrnn::IndRNN)(inp)
    state = zeros_like(inp, size(indrnn.cell.u, 1))
    return indrnn(inp, state)
end
  
function (indrnn::IndRNN)(inp, state) 
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = indrnn.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end