#https://arxiv.org/pdf/1803.10225
struct LiGRUCell{I, H, V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer LiGRUCell

"""
    LiGRUCell((in, out)::Pair; init = glorot_uniform, bias = true)
"""
function LiGRUCell((in, out)::Pair;
    init = glorot_uniform,
    bias = true)

    Wi = init(out * 2, in)
    Wh = init(out * 2, out)
    b = create_bias(Wi, bias, size(Wi, 1))

    return LiGRUCell(Wi, Wh, b)
end

LiGRUCell(in, out; kwargs...) = LiGRUCell(in => out; kwargs...)

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

"""
    LiGRU((in, out)::Pair; init = glorot_uniform, bias = true)
"""
function LiGRU((in, out)::Pair; init = glorot_uniform, bias = true)
    cell = LiGRUCell(in => out; init, bias)
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
