#function dense_proj(weight::AbstractMatrix, inp_or_state::AbstractVector, bias::AbstractVector)
#    return weight * inp_or_state .+ bias
#end

function dense_proj(weight::AbstractMatrix, inp_or_state::AbstractVector, bias::AbstractVector)
    weight_inp = weight * inp_or_state
    if bias isa AbstractVector
        if axes(bias) === axes(weight_inp)
            return weight_inp .+ bias
        else
            tmp = similar(weight_inp, eltype(bias))
            copyto!(tmp, bias)
            return weight_inp .+ tmp
        end
    else
        return weight_inp .+ bias
    end
end

#independent recurrence has only state since it's only for weight_hh
function dense_proj(weight::AbstractVector, state::AbstractVector, bias::AbstractVector)
    hidden_size = length(state)
    num_gates = div(length(weight), hidden_size)
    re_weight = reshape(weight, hidden_size, num_gates)
    re_bias = reshape(bias, hidden_size, num_gates)
    proj = @. re_weight * state + re_bias
    return vec(proj)
end

function add_projections(weight_b_ih::AbstractVector, weight_b_hh::AbstractVector)
    return weight_b_ih .+ weight_b_hh
end

function mul_projections(weight_b_ih::AbstractVector, weight_b_hh::AbstractVector)
    return weight_b_ih .* weight_b_hh
end
