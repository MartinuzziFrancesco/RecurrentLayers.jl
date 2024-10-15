struct LiGRUCell{I, H, V, S, F1, F2}
    Wf::I
    Wh::H
    b::V
    state0::S
    activation_fn::F1
    gate_activation_fn::F2
end

function LiGRUCell((in, out)::Pair;
    init=glorot_uniform,
    initb=zeros32,
    init_state=zeros32,
    activation_fn=tanh_fast,
    gate_activation_fn=sigmoid_fast)

    Wf = init(out * 2, in)
    Wh = init(out * 2, out)
    b = initb(out * 2)
    state0 = init_state(out, 1)
    return LiGRUCell(Wf, Wh, b, state0, activation_fn, gate_activation_fn)
end

LiGRUCell(in, out; kwargs...) = LiGRUCell(in => out; kwargs...)

function (ligru::LiGRUCell{I,H,V,<:AbstractMatrix{T},F1, F2})(hidden, inp::AbstractVecOrMat) where {I,H,V,T,F1,F2}
    _size_check(ligru, inp, 1 => size(ligru.Wf,2))
    Wf, Wh, bias, o = ligru.Wf, ligru.Wh, ligru.b, size(hidden, 1)
    inp_t = _match_eltype(ligru, T, inp)
    gxs, ghs, bs = multigate(Wf*inp_t, o, Val(2)), multigate(Wh*(hidden), o, Val(2)), multigate(bias, o, Val(2))
    forget_gate = @. ligru.gate_activation_fn(gxs[1] + ghs[1] + bs[1])

    candidate_hidden = @. tanh_fast(gxs[2] + ghs[2] + bs[2])
    new_h = forget_gate .* hidden .+ (1 .- forget_gate) .* candidate_hidden
    return new_h, reshape_cell_output(new_h, inp)
end

Flux.@layer LiGRUCell

Base.show(io::IO, ligru::LiGRUCell) =
    print(io, "LiGRUCell(", size(ligru.Wf, 2), " => ", size(ligru.Wf, 1) ÷ 2, ")")

function LiGRU(args...; kwargs...)
    return Flux.Recur(LiGRUCell(args...; kwargs...))
end    

function Flux.Recur(ligru::LiGRUCell)
    return Flux.Recur(ligru, ligru.state0)
end