#function dense_proj(weight::AbstractMatrix, inp_or_state::AbstractVector, bias::AbstractVector)
#    return weight * inp_or_state .+ bias
#end

function dense_proj(W::AbstractMatrix, x::AbstractVecOrMat, b::AbstractVector)
    y = W * x
    add_bias!(y, b)

    return y
end

function add_bias!(y::AbstractVector, b::AbstractVector)
    @assert length(y) == length(b)
    @inbounds for I in eachindex(y, b)
        y[I] += b[I]
    end
end

function add_bias!(y::AbstractMatrix, b::AbstractVector)
    @assert size(y, 1) == length(b)
    @inbounds for j in axes(y, 2), i in axes(y, 1)

        y[i, j] += b[i]
    end
    return y
end

function dense_proj(W::AbstractMatrix, x::AbstractVecOrMat, b::Bool)
    return W * x
end

#independent recurrence has only state since it's only for weight_hh
function dense_proj(weight::AbstractVector, state::AbstractVecOrMat, bias::AbstractVecOrMat)
    hidden_size = length(state)
    num_gates = div(length(weight), hidden_size)
    re_weight = reshape(weight, hidden_size, num_gates)
    re_bias = reshape(bias, hidden_size, num_gates)
    proj = @. re_weight * state + re_bias
    return vec(proj)
end

function add_projections(weight_b_ih::AbstractVecOrMat, weight_b_hh::AbstractVecOrMat)
    return weight_b_ih .+ weight_b_hh
end

function mul_projections(weight_b_ih::AbstractVecOrMat, weight_b_hh::AbstractVecOrMat)
    return weight_b_ih .* weight_b_hh
end
