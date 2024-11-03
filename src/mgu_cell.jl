#https://arxiv.org/pdf/1603.09420
struct MGUCell{I, H, V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer MGUCell

"""
    MGUCell((in, out)::Pair;
    kernel_init = glorot_uniform,
    recurrent_kernel_init = glorot_uniform,
    bias = true)
"""
function MGUCell((in, out)::Pair;
    kernel_init = glorot_uniform,
    recurrent_kernel_init = glorot_uniform,
    bias = true)

    Wi = kernel_init(out * 2, in)
    Wh = recurrent_kernel_init(out * 2, out)
    b = create_bias(Wi, bias, size(Wi, 1))

    return MGUCell(Wi, Wh, b)
end

MGUCell(in, out; kwargs...) = MGUCell(in => out; kwargs...)

function (mgu::MGUCell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(mgu.Wh, 2))
    return mgu(inp, state)
end

function (mgu::MGUCell)(inp::AbstractVecOrMat, state)
    _size_check(mgu, inp, 1 => size(mgu.Wi,2))
    Wi, Wh, b = mgu.Wi, mgu.Wh, mgu.bias
    #split
    gxs = chunk(Wi * inp .+ b, 2, dims=1)
    ghs = chunk(Wh, 2, dims=1)

    forget_gate = sigmoid_fast.(gxs[1] .+ ghs[1]*state)
    candidate_state = tanh_fast.(gxs[2] .+ ghs[2]*(forget_gate.*state))
    new_state = forget_gate .* state .+ (1 .- forget_gate) .* candidate_state
    return new_state
end

Base.show(io::IO, mgu::MGUCell) =
    print(io, "MGUCell(", size(mgu.Wi, 2), " => ", size(mgu.Wi, 1) ÷ 2, ")")


struct MGU{M}
    cell::M
end
  
Flux.@layer :expand MGU

"""
    MGU((in, out)::Pair; init = glorot_uniform, bias = true)
"""
function MGU((in, out)::Pair; init = glorot_uniform, bias = true)
    cell = MGUCell(in => out; init, bias)
    return MGU(cell)
end

function (mgu::MGU)(inp)
    state = zeros_like(inp, size(mgu.cell.Wh, 2))
    return mgu(inp, state)
end
  
function (mgu::MGU)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = mgu.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end
