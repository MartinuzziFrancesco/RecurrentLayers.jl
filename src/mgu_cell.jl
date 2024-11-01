#https://arxiv.org/pdf/1603.09420
struct MGUCell{I, H, V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer MGUCell

"""
    MGUCell((in, out)::Pair; init = glorot_uniform, bias = true)
"""
function MGUCell((in, out)::Pair;
    init = glorot_uniform,
    bias = true)

    Wi = init(out * 2, in)
    Wh = init(out * 2, out)
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
    cells::Vector{M}
    dropout::Real
end

Flux.@layer :expand MGU

"""
    MGU((in_size, out_size)::Pair; n_layers = 1, dropout = 0.0, init = glorot_uniform, bias = true)
"""
function MGU((in_size, out_size)::Pair;
    n_layers::Int=1,
    dropout::Float64=0.0,
    kwargs...)
    cells = []
    for i in 1:n_layers
        tin_size = i == 1 ? in_size : out_size
        push!(cells, MGUCell(tin_size => out_size; kwargs...))
    end
    return MGU(cells, dropout)
end

# Forward pass without initial state
function (mgu::MGU)(input)
    batch_size = size(input, 3)
    state = [zeros(size(mgu.cells[i].Wh, 2), batch_size) for i in 1:length(mgu.cells)]
    return mgu(input, state)
end

function (mgu::MGU)(inp, initial_states)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    num_layers = length(mgu.cells)
    foldl((acc, idx) -> begin
        (layer_input, states) = acc
        cell = mgu.cells[idx]
        layer_output, new_state = _process_layer(layer_input, states[idx], cell)
        updated_states = vcat(states[1:idx-1], [new_state], states[idx+1:end])
        return layer_output, updated_states
    end, 1:num_layers, init=(inp, initial_states))[1]
end
