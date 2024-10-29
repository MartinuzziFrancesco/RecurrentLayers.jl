#https://www.mdpi.com/2079-9292/13/16/3204
struct LightRUCell{I,H,V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer LightRUCell

function LightRUCell((in, out)::Pair, σ=tanh; init = glorot_uniform, bias = true)
    Wi = init(2 * out, in)
    Wh = init(out, out)
    b = create_bias(Wi, bias, size(Wh, 1))

    return LightRUCell(Wi, Wh, b)
end

LightRUCell(in, out; kwargs...) = LightRUCell(in => out; kwargs...)

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

function LightRU((in, out)::Pair; init = glorot_uniform, bias = true)
    cell = LightRUCell(in => out; init, bias)
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
