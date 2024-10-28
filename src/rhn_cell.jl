#https://arxiv.org/pdf/1607.03474
#https://github.com/jzilly/RecurrentHighwayNetworks/blob/master/rhn.py#L138C1-L180C60

struct RHNCellUnit{I,V}
    weight::I
    bias::V
end

function RHNCellUnit((in, out)::Pair; init = glorot_uniform, bias = true)
    weight = init(3 * out, in)
    b = create_bias(weight, bias, size(weight, 1))
    return RHNCellUnit(weight, b)
end

function (rhn::RHNCellUnit)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(rhn.weight, 2))
    return rhn(inp, state)
end

function (rhn::RHNCellUnit)(inp::AbstractVecOrMat, state)
    _size_check(rhn, inp, 1 => size(rhn.weight, 2))
    weight, bias = rhn.weight, rhn.bias

    #compute
    pre_nonlin = weight * inp + bias

    #split
    pre_h, pre_t, pre_c = chunk(pre_nonlin, 3, dims = 1)
    return pre_h, pre_t, pre_c
end

Base.show(io::IO, rhn::RHNCellUnit) =
    print(io, "RHNCellUnit(", size(rhn.weight, 2), " => ", size(rhn.weight, 1)÷3, ")")

struct RHNCell{C}
    layers::C
    couple_carry::Bool
end

function RHNCell((in, out), depth=3;
    couple_carry::Bool = true,
    cell_kwargs...)

    layers = []
    for layer in 1:depth
        if layer == 1
            real_in = in + out
        else
            real_in = out
        end
        rhn = RHNCellUnit(real_in=>out; cell_kwargs...)
        push!(layers, rhn)
    end
    return RHNCell(Chain(layers), couple_carry)
end


function (rhn::RHNCell)(inp, state=nothing)

    #not ideal
    if state == nothing
        state = zeros_like(inp, size(rhn.layers.layers[2].weight, 2))
    end

    current_state = state

    for (i, layer) in enumerate(rhn.layers.layers)
        if i == 1
            inp_combined = vcat(inp, current_state)
        else
            inp_combined = current_state
        end

        pre_h, pre_t, pre_c = layer(inp_combined)

        # Apply nonlinearities
        hidden_gate = tanh.(pre_h)
        transform_gate = sigmoid.(pre_t)
        carry_gate = sigmoid.(pre_c)

        # Highway component
        if rhn.couple_carry
            current_state = (hidden_gate .- current_state) .* transform_gate .+ current_state
        else
            current_state = hidden_gate .* transform_gate .+ current_state .* carry_gate
        end
    end

    return current_state
end
