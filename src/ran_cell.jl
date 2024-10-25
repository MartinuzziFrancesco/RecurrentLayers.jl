#https://arxiv.org/pdf/1705.07393
struct RANCell{I,H,V}
    Wi::I
    Wh::H
    bias::V
end

function RANCell((in, out)::Pair, σ=tanh; init = glorot_uniform, bias = true)
    Wi = init(3 * out, in)
    Wh = init(2 * out, out)
    b = create_bias(Wi, bias, size(Wh, 1))

    return RANCell(Wi, Wh, b)
end

RANCell(in, out; kwargs...) = RANCell(in => out; kwargs...)

function (ran::RANCell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(ran.Wh, 2))
    c_state = zeros_like(state)
    return ran(inp, (state, c_state))
end

function (ran::RANCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(ran, inp, 1 => size(ran.Wi,2))
    Wi, Wh, b = ran.Wi, ran.Wh, ran.bias

    #split
    gxs = chunk(Wi * inp, 3, dims=1)
    bs = chunk(b, 2, dims=1)
    ghs = chunk(Wh * state, 2, dims=1)

    #compute
    input_gate = @. sigmoid_fast(gxs[2] + ghs[1] + bs[1])
    forget_gate = @. sigmoid_fast(gxs[3] + ghs[2] + bs[2])
    candidate_state = @. input_gate * gxs[1] + forget_gate * c_state
    new_state = tanh_fast(candidate_state)
    return new_state, candidate_state
end

Base.show(io::IO, ran::RANCell) =
    print(io, "RANCell(", size(ran.Wi, 2), " => ", size(ran.Wi, 1)÷3, ")")


struct RAN{M}
    cell::M
end

Flux.@layer :expand RAN

function RAN((in, out)::Pair; init = glorot_uniform, bias = true)
    cell = RANCell(in => out; init, bias)
    return RAN(cell)
end

function (ran::RAN)(inp)
    state = zeros_like(inp, size(ran.cell.Wh, 2))
    c_state = zeros_like(state)
    return ran(inp, (state, c_state))
end

function (ran::RAN)(inp, (state, c_state))
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    new_cstate = []
    for inp_t in eachslice(inp, dims=2)
        state, c_state = ran.cell(inp_t, (state, c_state))
        new_state = vcat(new_state, [state])
        new_cstate = vcat(new_cstate, [c_state])
    end
    return stack(new_state, dims=2), stack(new_cstate, dims=2)
end

