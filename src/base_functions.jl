#function dense_proj(weight::AbstractMatrix, inp_or_state::AbstractVector, bias::AbstractVector)
#    return weight * inp_or_state .+ bias
#end

function dense_proj(weight::AbstractMatrix, inp_or_state::AbstractVecOrMat, bias::Bool)
    return weight * inp_or_state
end

function dense_proj(
        weight::AbstractMatrix, inp_or_state::AbstractVecOrMat, bias::AbstractVector)
    weight_inporstate = dense_proj(weight, inp_or_state, false)
    add_bias!(weight_inporstate, bias)
    return weight_inporstate
end

function add_bias!(weight_inporstate::AbstractVector, bias::AbstractVector)
    @assert length(weight_inporstate) == length(bias)
    @inbounds for idx in eachindex(weight_inporstate, bias)
        weight_inporstate[idx] += bias[idx]
    end
end

function add_bias!(weight_inporstate::AbstractMatrix, bias::AbstractVector)
    @assert size(weight_inporstate, 1) == length(bias)
    @inbounds for jdx in axes(weight_inporstate, 2), idx in axes(weight_inporstate, 1)

        weight_inporstate[idx, jdx] += bias[idx]
    end
    return weight_inporstate
end

#independent recurrence has only state since it's only for weight_hh
function dense_proj(
        weight::AbstractVector, state::AbstractVector, bias::Union{
            AbstractVector, Bool})
    proj = _ind_rec(weight, state, bias)
    return proj
end

function dense_proj(
        weight::AbstractVector, state::AbstractMatrix, bias::Union{
            AbstractVector, Bool})
    return _ind_rec(weight, state, bias)
end

function _ind_rec(weight::AbstractVector, state::AbstractVecOrMat, bias::AbstractVector)
    hidden_size = size(state, 1)
    weight_size = length(weight)
    num_gates = div(weight_size, hidden_size)

    if ndims(state) == 1
        re_weight = reshape(weight, hidden_size, num_gates)
        re_bias = reshape(bias, hidden_size, num_gates)
        proj = re_weight .* state .+ re_bias
        return vec(proj)
    else
        batch_size = size(state, 2)
        proj = reshape(weight, hidden_size, num_gates, 1) .*
               reshape(state, hidden_size, 1, batch_size) .+
               reshape(bias, hidden_size, num_gates, 1)
        return reshape(proj, hidden_size * num_gates, batch_size)
    end
end

function _ind_rec(weight::AbstractVector, state::AbstractVecOrMat, bias::Bool)
    hidden_size = size(state, 1)
    num_gates = div(length(weight), hidden_size)
    re_weight = reshape(weight, hidden_size, num_gates)
    proj = @. re_weight * state
    return proj
end

function add_projections(weight_b_ih::AbstractVecOrMat, weight_b_hh::AbstractVecOrMat)
    return weight_b_ih .+ weight_b_hh
end

function mul_projections(weight_b_ih::AbstractVecOrMat, weight_b_hh::AbstractVecOrMat)
    return weight_b_ih .* weight_b_hh
end
