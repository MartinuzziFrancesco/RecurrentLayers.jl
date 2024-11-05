#https://proceedings.mlr.press/v37/jozefowicz15.pdf
struct MUT1Cell{I, H, V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer MUT1Cell

"""
    MUT1Cell((in, out)::Pair;
        kernel_init = glorot_uniform,
        recurrent_kernel_init = glorot_uniform,
        bias = true)
"""
function MUT1Cell((in, out)::Pair;
    kernel_init = glorot_uniform,
    recurrent_kernel_init = glorot_uniform,
    bias = true)

    Wi = kernel_init(out * 3, in)
    Wh = recurrent_kernel_init(out * 2, out)
    b = create_bias(Wi, bias, 3 * out)

    return MUT1Cell(Wi, Wh, b)
end

function (mut::MUT1Cell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(mut.Wh, 2))
    return mut(inp, state)
end

function (mut::MUT1Cell)(inp::AbstractVecOrMat, state)
    _size_check(mut, inp, 1 => size(mut.Wi,2))
    Wi, Wh, b = mut.Wi, mut.Wh, mut.bias
    #split
    gxs = chunk(Wi * inp, 3, dims=1)
    ghs = chunk(Wh, 2, dims=1)
    bs = chunk(b, 3, dims=1)

    forget_gate = sigmoid_fast.(gxs[1] .+ bs[1])
    reset_gate = sigmoid_fast.(gxs[2] .+ ghs[1]*state .+ bs[2])
    candidate_state = tanh_fast.(
        ghs[2] * (reset_gate .* state) .+ tanh_fast(gxs[3]) .+ bs[3]
    ) #in the paper is tanh(x_t) but dimensionally it cannot work
    new_state = candidate_state .* forget_gate .+ state .* (1 .- forget_gate)
    return new_state
end

Base.show(io::IO, mut::MUT1Cell) =
    print(io, "MUT1Cell(", size(mut.Wi, 2), " => ", size(mut.Wi, 1) ÷ 3, ")")

struct MUT1{M}
    cell::M
end
  
Flux.@layer :expand MUT1

"""
    MUT1((in, out)::Pair; kwargs...)
"""
function MUT1((in, out)::Pair; kwargs...)
    cell = MUT1Cell(in => out; kwargs...)
    return MUT1(cell)
end

function (mut::MUT1)(inp)
    state = zeros_like(inp, size(mut.cell.Wh, 2))
    return mut(inp, state)
end
  
function (mut::MUT1)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = mut.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end



struct MUT2Cell{I, H, V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer MUT2Cell

"""
    MUT2Cell((in, out)::Pair;
        kernel_init = glorot_uniform,
        recurrent_kernel_init = glorot_uniform,
        bias = true)
"""
function MUT2Cell((in, out)::Pair;
    kernel_init = glorot_uniform,
    recurrent_kernel_init = glorot_uniform,
    bias = true)

    Wi = kernel_init(out * 3, in)
    Wh = recurrent_kernel_init(out * 3, out)
    b = create_bias(Wi, bias, 3 * out)

    return MUT2Cell(Wi, Wh, b)
end

function (mut::MUT2Cell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(mut.Wh, 2))
    return mut(inp, state)
end

function (mut::MUT2Cell)(inp::AbstractVecOrMat, state)
    _size_check(mut, inp, 1 => size(mut.Wi,2))
    Wi, Wh, b = mut.Wi, mut.Wh, mut.bias
    #split
    gxs = chunk(Wi * inp, 3, dims=1)
    ghs = chunk(Wh, 3, dims=1)
    bs = chunk(b, 3, dims=1)

    forget_gate = sigmoid_fast.(gxs[1] .+ ghs[1] * state .+ bs[1])
    # the dimensionlity alos does not work here like the paper describes it
    reset_gate = sigmoid_fast.(gxs[2] .+ ghs[2]*state .+ bs[2]) 
    candidate_state = tanh_fast.(ghs[3] * (reset_gate .* state) .+ gxs[3] .+ bs[3])
    new_state = candidate_state .* forget_gate .+ state .* (1 .- forget_gate)
    return new_state
end

Base.show(io::IO, mut::MUT2Cell) =
    print(io, "MUT2Cell(", size(mut.Wi, 2), " => ", size(mut.Wi, 1) ÷ 3, ")")


struct MUT2{M}
    cell::M
end
  
Flux.@layer :expand MUT2

"""
    MUT1Cell((in, out)::Pair; kwargs...)
"""
function MUT2((in, out)::Pair; kwargs...)
    cell = MUT2Cell(in => out; kwargs...)
    return MUT2(cell)
end

function (mut::MUT2)(inp)
    state = zeros_like(inp, size(mut.cell.Wh, 2))
    return mut(inp, state)
end
  
function (mut::MUT2)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = mut.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end


struct MUT3Cell{I, H, V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer MUT3Cell

"""
    MUT3Cell((in, out)::Pair;
        kernel_init = glorot_uniform,
        recurrent_kernel_init = glorot_uniform,
        bias = true)
"""
function MUT3Cell((in, out)::Pair;
    kernel_init = glorot_uniform,
    recurrent_kernel_init = glorot_uniform,
    bias = true)

    Wi = kernel_init(out * 3, in)
    Wh = recurrent_kernel_init(out * 3, out)
    b = create_bias(Wi, bias, 3 * out)

    return MUT3Cell(Wi, Wh, b)
end

function (mut::MUT3Cell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(mut.Wh, 2))
    return mut(inp, state)
end

function (mut::MUT3Cell)(inp::AbstractVecOrMat, state)
    _size_check(mut, inp, 1 => size(mut.Wi,2))
    Wi, Wh, b = mut.Wi, mut.Wh, mut.bias
    #split
    gxs = chunk(Wi * inp, 3, dims=1)
    ghs = chunk(Wh, 3, dims=1)
    bs = chunk(b, 3, dims=1)

    forget_gate = sigmoid_fast.(gxs[1] .+ ghs[1] * tanh_fast(state) .+ bs[1])
    reset_gate = sigmoid_fast.(gxs[2] .+ ghs[2]*state .+ bs[2])
    candidate_state = tanh_fast.(ghs[3] * (reset_gate .* state) .+ gxs[3] .+ bs[3])
    new_state = candidate_state .* forget_gate .+ state .* (1 .- forget_gate)
    return new_state
end

Base.show(io::IO, mut::MUT3Cell) =
    print(io, "MUT3Cell(", size(mut.Wi, 2), " => ", size(mut.Wi, 1) ÷ 3, ")")

struct MUT3{M}
    cell::M
end
  
Flux.@layer :expand MUT3

"""
    MUT3((in, out)::Pair; kwargs...)
"""
function MUT3((in, out)::Pair; kwargs...)
    cell = MUT3Cell(in => out; kwargs...)
    return MUT3(cell)
end

function (mut::MUT3)(inp)
    state = zeros_like(inp, size(mut.cell.Wh, 2))
    return mut(inp, state)
end
  
function (mut::MUT3)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = mut.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end