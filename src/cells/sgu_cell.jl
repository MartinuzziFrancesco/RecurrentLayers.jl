"""
"""
struct SGUCell{A, B, C, D, F} <: AbstractRecurrentCell
    weight_ih::A
    weight_hh::B
    bias_ih::C
    bias_hh::D
    integration_fn::F
end

function SGUCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(2 * hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 2)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)
    return SGUCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (sgucell::SGUCell)(inp, state)
    _size_check(sgucell, inp, 1 => size(sgucell.weight_ih, 2))
    proj_ih = dense_proj(sgucell.weight_ih, inp, sgucell.bias_ih)
    gxs = chunk(proj_ih, 2; dims=1)
    whs = chunk(sgucell.weight_hh, 2; dims=1)
    bhs = chunk(sgucell.bias_hh, 2; dims=1)
    zg = tanh_fast.(whs[1] * (gxs[1] .* state) .+ bhs[1])
    zout = softplus.(zg .* state)
    zt_c1 = whs[2] * state .+ bhs[2]
    zt = hardsigmoid.(sgucell.integration_fn(gxs[2], zt_c1))
    new_state = @. (1 - zt) * state + zt * zout
    return new_state, new_state
end

function initialstates(sgucell::SGUCell)
    state = zeros_like(sgucell.weight_hh, size(sgucell.weight_hh, 1) ÷ 2)
    return state
end

function Base.show(io::IO, sgucell::SGUCell)
    print(io, "SGUCell(", size(sgucell.weight_ih, 2),
        " => ", size(sgucell.weight_ih, 1) ÷ 2, ")")
end


"""
"""
struct SGU{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand SGU

function SGU((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = SGUCell(input_size => hidden_size; kwargs...)
    return SGU{return_state, typeof(cell)}(cell)
end

function functor(rnn::SGU{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> SGU{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, sgu::SGU)
    print(io, "SGU(", size(sgu.cell.weight_ih, 2),
        " => ", size(sgu.cell.weight_ih, 1) ÷ 2)
    print(io, ")")
end