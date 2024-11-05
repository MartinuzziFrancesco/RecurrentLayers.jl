#https://arxiv.org/pdf/1705.07393
struct RANCell{I,H,V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer RANCell


"""
    RANCell((in, out)::Pair;
        kernel_init = glorot_uniform,
        recurrent_kernel_init = glorot_uniform,
        bias = true)

The `RANCell`, introduced in [this paper](https://arxiv.org/pdf/1705.07393), 
is a recurrent cell layer which provides additional memory through the
use of gates.

and returns both h_t anf c_t.

See [`RAN`](@ref) for a layer that processes entire sequences.

# Arguments

- `in => out`: Specifies the input and output dimensions of the layer.
- `init`: Initialization function for the weight matrices, default is `glorot_uniform`.
- `bias`: Indicates if a bias term is included; the default is `true`.

# Forward

    rancell(x, [h, c])

The forward pass takes the following arguments:

- `x`: Input to the cell, which can be a vector of size `in` or a matrix of size `in x batch_size`.
- `h`: The hidden state vector of the cell, sized `out`, or a matrix of size `out x batch_size`.
- `c`: The candidate state, sized `out`, or a matrix of size `out x batch_size`.
If not provided, both `h` and `c` default to vectors of zeros.

# Examples

```julia
rancell = RANCell(3 => 5)
inp = rand(Float32, 3)
#initializing the hidden states, if we want to provide them
state = rand(Float32, 5)
c_state = rand(Float32, 5)

#result with default initialization of internal states
result = rancell(inp)
#result with internal states provided
result_state = rancell(inp, (state, c_state))
```
"""
function RANCell((in, out)::Pair;
    kernel_init = glorot_uniform,
    recurrent_kernel_init = glorot_uniform,
    bias = true)
    Wi = kernel_init(3 * out, in)
    Wh = recurrent_kernel_init(2 * out, out)
    b = create_bias(Wi, bias, size(Wh, 1))
    return RANCell(Wi, Wh, b)
end

RANCell(in, out; kwargs...) = RANCell(in => out; kwargs...)

function (ran::RANCell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(ran.Wh, 2))
    c_state = zeros_like(state)
    return ran(inp, (state, c_state))
end

function (ran::RANCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(ran, inp, 1 => size(ran.Wi,2))
    Wi, Wh, b = ran.Wi, ran.Wh, ran.bias

    #split
    gxs = chunk(Wi * inp, 3; dims=1)
    ghs = chunk(Wh * state .+ b, 2; dims=1)

    #compute
    input_gate = @. sigmoid_fast(gxs[2] + ghs[1])
    forget_gate = @. sigmoid_fast(gxs[3] + ghs[2])
    candidate_state = @. input_gate * gxs[1] + forget_gate * c_state
    new_state = tanh_fast(candidate_state)
    return new_state, candidate_state
end

Base.show(io::IO, ran::RANCell) =
    print(io, "RANCell(", size(ran.Wi, 2), " => ", size(ran.Wi, 1)÷3, ")")


struct RAN{M}
    cell::M
end

Flux.@layer :expand RAN

"""
    RAN(in => out; kwargs...)

"""
function RAN((in, out)::Pair; kwargs...)
    cell = RANCell(in => out; kwargs...)
    return RAN(cell)
end

function (ran::RAN)(inp)
    state = zeros_like(inp, size(ran.cell.Wh, 2))
    c_state = zeros_like(state)
    return ran(inp, (state, c_state))
end

function (ran::RAN)(inp, (state, c_state))
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    new_cstate = []
    for inp_t in eachslice(inp, dims=2)
        state, c_state = ran.cell(inp_t, (state, c_state))
        new_state = vcat(new_state, [state])
        new_cstate = vcat(new_cstate, [c_state])
    end
    return stack(new_state, dims=2), stack(new_cstate, dims=2)
end

