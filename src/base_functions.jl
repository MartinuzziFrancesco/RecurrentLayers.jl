function dense_proj(weight::AbstractMatrix, inp_or_state::AbstractVector, bias::AbstractVector)
    return weight * inp_or_state .+ bias
end

function dense_proj(weight::AbstractVector, inp_or_state::AbstractVector, bias::AbstractVector)
    return @. weight * inp_or_state + bias
end

function add_projections(weight_b_ih::AbstractVector, weight_b_hh::AbstractVector)
    return weight_b_ih .+ weight_b_hh
end

function mul_projections(weight_b_ih::AbstractVector, weight_b_hh::AbstractVector)
    return weight_b_ih .* weight_b_hh
end
