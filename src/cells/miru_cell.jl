
"""
"""
struct MiRU1Cell{I, H, V, W, L, F, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    update_coefficient::L
    activation::F
    integration_fn::A
end

function MiRU1Cell((input_size, hidden_size)::Pair{<:Int, <:Int}, activation=tanh;
    init_kernel = glorot_uniform, init_recurrent_kernel = glorot_uniform,
    bias::Bool=true, recurrent_bias::Bool=true,
    update_coefficient::AbstractFloat = 0.5,
    integration_mode::Symbol=:addition,
    independent_recurrence::Bool=false)

    weight_ih = init_kernel(2 * hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, 2 * hidden_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    _update_coefficient = eltype(weight_ih)(update_coefficient)
    integration_fn = _integration_fn(integration_mode)
    
    return MiRU1Cell(weight_ih, weight_hh, bias_ih, bias_hh, _update_coefficient, activation, integration_fn)
end

function (miru::MiRUCell)(inp::AbstractVecOrMat, state)
    _size_check(miru, inp, 1 => size(miru.weight_ih, 2))
    proj_ih = dense_proj(miru.weight_ih, inp, bias_ih)
    gxs = chunk(proj_ih, 2; dims=1)
    chunk_whh = chunk(weight_hh, 2; dims=1)
    chunk_bhh = chunk(bias_hh, 2; dims=1)
    rec_r = dense_proj(chunk_whh[1], state, chunk_bhh[1])
    first_gate = miru.activation.(miru.integration_fn(gxs[1], rec_r))
    rec_h = dense_proj(chunk_whh[2], first_gate .* state, chunk_bhh[2])
    candidate_state = tanh_fast.(miru.integration_fn(gxs[2], rec_h))
    new_state = miru.update_coefficient .* state .+ (1 - miru.update_coefficient) .* candidate_state

    return new_state, new_state
end

