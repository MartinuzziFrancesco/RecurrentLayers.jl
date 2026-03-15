"""
    IntersectionRNNCell
"""
struct IntersectionRNNCell{I, H, V, W, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

function IntersectionRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition, independent_recurrence::Bool=true)

    weight_ih = init_kernel(hidden_size * 4, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 4)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)

    return IntersectionRNNCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (irnn::IntersectionRNNCell)(inp::AbstractVecOrMat, state)
    _size_check(irnn, inp, 1 => size(irnn.weight_ih, 2))
    proj_ih = dense_proj(irnn.weight_ih, inp, irnn.bias_ih)
    proj_hh = dense_proj(irnn.weight_hh, state, irnn.bias_hh)
    gxs = chunk(proj_ih, 4; dims=1)
    ghs = chunk(proj_hh, 4; dims=1)
    yin = relu.(irnn.integration_fn(gxs[1], ghs[1]))
    hin = tanh_fast.(irnn.integration_fn(gxs[2], ghs[2]))
    gy = sigmoid_fast.(irnn.integration_fn(gxs[3], ghs[3]))
    gh = sigmoid_fast.(irnn.integration_fn(gxs[4], ghs[4]))
    new_inp = gy .* inp .+ (1 .- gh) .* yin
    new_state = gh .* state .+ (1 .- gh) .* hin

    return new_inp, new_state ## double check thi
end

function initialstates(irnn::IntersectionRNNCell)
    return zeros_like(irnn.weight_hh, size(irnn.weight_hh, 1) ÷ 4)
end

function Base.show(io::IO, irnn::IntersectionRNNCell)
    print(io, "IntersectionRNNCell(", size(irnn.weight_ih, 2), " => ", size(irnn.weight_ih, 1) ÷ 4, ")")
end

"""
"""
struct IntersectionRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand IntersectionRNN

function IntersectionRNN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = IntersectionRNNCell(input_size => hidden_size; kwargs...)
    return IntersectionRNN{return_state, typeof(cell)}(cell)
end

function functor(rnn::IntersectionRNN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> IntersectionRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, irnn::IntersectionRNN)
    print(io, "IntersectionRNN(", size(irnn.cell.weight_ih, 2), " => ", size(irnn.cell.bias_ih, 1) ÷ 4)
    print(io, ")")
end