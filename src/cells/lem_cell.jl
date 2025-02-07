#https://arxiv.org/pdf/2110.04744
@doc raw"""
    LEMCell(input_size => hidden_size, [dt];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Long expressive memory unit](https://arxiv.org/pdf/2110.04744).
See [`LEM`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `dt`: timestep. Defaul is 1.0

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}

\end{aligned}
```

# Forward

    lemcell(inp, state)
    lemcell(inp)

## Arguments
- `inp`: The input to the lemcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the LEMCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct LEMCell{I, H, V, D} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    bias::V
    dt::D
end

@layer LEMCell

function LEMCell((input_size, hidden_size)::Pair{<:Int, <:Int}, dt::Number=1.0;
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size * 4, input_size)
    Wh = init_recurrent_kernel(hidden_size * 3, hidden_size)
    Wz = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))

    return LEMCell(Wi, Wh, b)
end

function (lem::LEMCell)(inp::AbstractVecOrMat, (state, z_state))
    _size_check(lem, inp, 1 => size(lem.Wi, 2))
    Wi, Wh, b = lem.Wi, lem.Wh, lem.bias
    #split
    gxs = chunk(Wi * inp .+ b, 4; dims=1)
    ghs = chunk(Wh * state, 3; dims=1)

    msdt_bar = lem.dt .* sigmoid_fast.(gxs[1] .+ ghs[1])
    ms_dt = lem.dt .* sigmoid_fast.(gxs[2] .+ ghs[2])
    new_zstate = (1.0 .- ms_dt) .* z_state .+ ms_dt .* tanh_fast(gxs[3] .+ ghs[3])
    new_state = (1.0 .- msdt_bar) .* state .+ msdt_bar .* tanh_fast(gxs[4] .+ Wz*z_state)
    return new_zstate, (new_state, new_zstate)
end

function Base.show(io::IO, lem::LEMCell)
    print(io, "LEMCell(", size(lem.Wi, 2), " => ", size(lem.Wi, 1) ÷ 2, ")")
end