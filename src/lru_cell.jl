#https://www.mdpi.com/2079-9292/13/16/3204
struct LRUCell{I,H,V}
    Wi::I
    Wh::H
    bias::V
end

function LRUCell((in, out)::Pair, σ=tanh; init = glorot_uniform, bias = true)
    Wi = init(2 * out, in)
    Wh = init(out, out)
    b = create_bias(Wi, bias, size(Wh, 1))

    return LRUCell(Wi, Wh, b)
end

LRUCell(in, out; kwargs...) = LRUCell(in => out; kwargs...)

function (lru::LRUCell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(lru.Wh, 2))
    return lru(inp, state)
end

function (lru::LRUCell)(inp::AbstractVecOrMat, state)
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

Base.show(io::IO, lru::LRUCell) =
    print(io, "LRUCell(", size(lru.Wi, 2), " => ", size(lru.Wi, 1)÷2, ")")



struct LRU{M}
    cell::M
end
  
Flux.@layer :expand LRU

function LRU((in, out)::Pair; init = glorot_uniform, bias = true)
    cell = LRUCell(in => out; init, bias)
    return LRU(cell)
end
  
function (lru::LRU)(inp)
    state = zeros_like(inp, size(lru.cell.Wh, 2))
    return lru(inp, state)
end
  
function (lru::LRU)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = lru.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end
