struct IndRNNCell{F,I,H,V}
    σ::F
    Wi::I
    u::H
    b::V
end

Flux.@layer IndRNNCell

function IndRNNCell((in, out)::Pair, σ=relu; init = glorot_uniform, bias = true)
    Wi = init(out, in)
    u = init(out)
    b = create_bias(Wi, bias, size(Wi, 1))
    return IndRNNCell(σ, Wi, u, b)
end

function (indrnn::IndRNNCell)(x::AbstractVecOrMat)
    state = zeros_like(x, size(indrnn.u, 1))
    return indrnn(x, state)

function (indrnn::IndRNNCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(indrnn, inp, 1 => size(indrnn.Wi, 2))
    σ = NNlib.fast_act(indrnn.σ, inp)
    state = σ.(indrnn.Wi*inp .+ indrnn.u.*state .+ indrnn.b)
    return state
end

function Base.show(io::IO, m::IndRNNCell)
    print(io, "IndRNNCell(", size(m.Wi, 2), " => ", size(indrnn.Wi, 1))
    print(io, ", ", indrnn.σ)
    print(io, ")")
end

struct IndRNN{M}
    cell::M
end
  
Flux.@layer :expand IndRNN
  
function IndRNN((in, out)::Pair, σ = tanh; bias = true, init = glorot_uniform)
    cell = IndRNNCell(in => out, σ; bias=bias, init=init)
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