#https://proceedings.mlr.press/v37/jozefowicz15.pdf
struct MUT1Cell{I, H, V} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer MUT1Cell

@doc raw"""
    MUT1Cell((input_size => hidden_size);
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Mutated unit 1 cell](https://proceedings.mlr.press/v37/jozefowicz15.pdf).
See [`MUT1`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
z &= \sigma(W_z x_t + b_z), \\
r &= \sigma(W_r x_t + U_r h_t + b_r), \\
h_{t+1} &= \tanh(U_h (r \odot h_t) + \tanh(W_h x_t) + b_h) \odot z \\
&\quad + h_t \odot (1 - z).
\end{aligned}
```

# Forward

    mutcell(inp, state)
    mutcell(inp)

## Arguments
- `inp`: The input to the mutcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the MUTCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state `new_state`, 
a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
function MUT1Cell((input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias::Bool = true)

    Wi = init_kernel(hidden_size * 3, input_size)
    Wh = init_recurrent_kernel(hidden_size * 2, hidden_size)
    b = create_bias(Wi, bias, 3 * hidden_size)

    return MUT1Cell(Wi, Wh, b)
end

function (mut::MUT1Cell)(inp::AbstractVecOrMat, state)
    _size_check(mut, inp, 1 => size(mut.Wi,2))
    Wi, Wh, b = mut.Wi, mut.Wh, mut.bias
    #split
    gxs = chunk(Wi * inp .+ b, 3, dims=1)
    ghs = chunk(Wh, 2, dims=1)

    forget_gate = sigmoid_fast.(gxs[1])
    reset_gate = sigmoid_fast.(gxs[2] .+ ghs[1]*state)
    candidate_state = tanh_fast.(
        ghs[2] * (reset_gate .* state) .+ tanh_fast(gxs[3])
    ) #in the paper is tanh(x_t) but dimensionally it cannot work
    new_state = candidate_state .* forget_gate .+ state .* (1 .- forget_gate)
    return new_state, new_state
end

Base.show(io::IO, mut::MUT1Cell) =
    print(io, "MUT1Cell(", size(mut.Wi, 2), " => ", size(mut.Wi, 1) ÷ 3, ")")

struct MUT1{M} <: AbstractRecurrentLayer
    cell::M
end
  
Flux.@layer :noexpand MUT1

@doc raw"""
    MUT1((input_size => hidden_size); kwargs...)

[Mutated unit 1 network](https://proceedings.mlr.press/v37/jozefowicz15.pdf).
See [`MUT1Cell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
z &= \sigma(W_z x_t + b_z), \\
r &= \sigma(W_r x_t + U_r h_t + b_r), \\
h_{t+1} &= \tanh(U_h (r \odot h_t) + \tanh(W_h x_t) + b_h) \odot z \\
&\quad + h_t \odot (1 - z).
\end{aligned}
```

# Forward

    mut(inp, state)
    mut(inp)

## Arguments
- `inp`: The input to the mut. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the MUT. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
"""
function MUT1((input_size, hidden_size)::Pair; kwargs...)
    cell = MUT1Cell(input_size => hidden_size; kwargs...)
    return MUT1(cell)
end


struct MUT2Cell{I, H, V}  <: AbstractRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer MUT2Cell

@doc raw"""
    MUT2Cell((input_size => hidden_size);
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Mutated unit 2 cell](https://proceedings.mlr.press/v37/jozefowicz15.pdf).
See [`MUT2`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
z &= \sigma(W_z x_t + U_z h_t + b_z), \\
r &= \sigma(x_t + U_r h_t + b_r), \\
h_{t+1} &= \tanh(U_h (r \odot h_t) + W_h x_t + b_h) \odot z \\
&\quad + h_t \odot (1 - z).
\end{aligned}
```

# Forward

    mutcell(inp, state)
    mutcell(inp)

## Arguments
- `inp`: The input to the mutcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the MUTCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state `new_state`, 
a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
function MUT2Cell((input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias::Bool = true)

    Wi = init_kernel(hidden_size * 3, input_size)
    Wh = init_recurrent_kernel(hidden_size * 3, hidden_size)
    b = create_bias(Wi, bias, 3 * hidden_size)

    return MUT2Cell(Wi, Wh, b)
end

function (mut::MUT2Cell)(inp::AbstractVecOrMat, state)
    _size_check(mut, inp, 1 => size(mut.Wi,2))
    Wi, Wh, b = mut.Wi, mut.Wh, mut.bias
    #split
    gxs = chunk(Wi * inp .+ b, 3, dims=1)
    ghs = chunk(Wh, 3, dims=1)

    forget_gate = sigmoid_fast.(gxs[1] .+ ghs[1] * state)
    # the dimensionlity alos does not work here like the paper describes it
    reset_gate = sigmoid_fast.(gxs[2] .+ ghs[2]*state) 
    candidate_state = tanh_fast.(ghs[3] * (reset_gate .* state) .+ gxs[3])
    new_state = candidate_state .* forget_gate .+ state .* (1 .- forget_gate)
    return new_state, new_state
end

Base.show(io::IO, mut::MUT2Cell) =
    print(io, "MUT2Cell(", size(mut.Wi, 2), " => ", size(mut.Wi, 1) ÷ 3, ")")


struct MUT2{M} <: AbstractRecurrentLayer
    cell::M
end
  
Flux.@layer :noexpand MUT2

@doc raw"""
    MUT2Cell((input_size => hidden_size); kwargs...)

[Mutated unit 2 network](https://proceedings.mlr.press/v37/jozefowicz15.pdf).
See [`MUT2Cell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
z &= \sigma(W_z x_t + U_z h_t + b_z), \\
r &= \sigma(x_t + U_r h_t + b_r), \\
h_{t+1} &= \tanh(U_h (r \odot h_t) + W_h x_t + b_h) \odot z \\
&\quad + h_t \odot (1 - z).
\end{aligned}
```

# Forward

    mut(inp, state)
    mut(inp)

## Arguments
- `inp`: The input to the mut. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the MUT. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
"""
function MUT2((input_size, hidden_size)::Pair; kwargs...)
    cell = MUT2Cell(input_size => hidden_size; kwargs...)
    return MUT2(cell)
end


struct MUT3Cell{I, H, V} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer MUT3Cell

@doc raw"""
    MUT3Cell((input_size => hidden_size);
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Mutated unit 3 cell](https://proceedings.mlr.press/v37/jozefowicz15.pdf).
See [`MUT3`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
z &= \sigma(W_z x_t + U_z \tanh(h_t) + b_z), \\
r &= \sigma(W_r x_t + U_r h_t + b_r), \\
h_{t+1} &= \tanh(U_h (r \odot h_t) + W_h x_t + b_h) \odot z \\
&\quad + h_t \odot (1 - z).
\end{aligned}
```

# Forward

    mutcell(inp, state)
    mutcell(inp)

## Arguments
- `inp`: The input to the mutcell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the MUTCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state `new_state`, 
a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
function MUT3Cell((input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true)

    Wi = init_kernel(hidden_size * 3, input_size)
    Wh = init_recurrent_kernel(hidden_size * 3, hidden_size)
    b = create_bias(Wi, bias, 3 * hidden_size)

    return MUT3Cell(Wi, Wh, b)
end

function (mut::MUT3Cell)(inp::AbstractVecOrMat, state)
    _size_check(mut, inp, 1 => size(mut.Wi,2))
    Wi, Wh, b = mut.Wi, mut.Wh, mut.bias
    #split
    gxs = chunk(Wi * inp .+ b, 3, dims=1)
    ghs = chunk(Wh, 3, dims=1)

    forget_gate = sigmoid_fast.(gxs[1] .+ ghs[1] * tanh_fast(state))
    reset_gate = sigmoid_fast.(gxs[2] .+ ghs[2]*state)
    candidate_state = tanh_fast.(ghs[3] * (reset_gate .* state) .+ gxs[3])
    new_state = candidate_state .* forget_gate .+ state .* (1 .- forget_gate)
    return new_state, new_state
end

Base.show(io::IO, mut::MUT3Cell) =
    print(io, "MUT3Cell(", size(mut.Wi, 2), " => ", size(mut.Wi, 1) ÷ 3, ")")

struct MUT3{M} <: AbstractRecurrentLayer
    cell::M
end
  
Flux.@layer :noexpand MUT3

@doc raw"""
    MUT3((input_size => hidden_size); kwargs...)

[Mutated unit 3 network](https://proceedings.mlr.press/v37/jozefowicz15.pdf).
See [`MUT3Cell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
z &= \sigma(W_z x_t + U_z \tanh(h_t) + b_z), \\
r &= \sigma(W_r x_t + U_r h_t + b_r), \\
h_{t+1} &= \tanh(U_h (r \odot h_t) + W_h x_t + b_h) \odot z \\
&\quad + h_t \odot (1 - z).
\end{aligned}
```

# Forward

    mut(inp, state)
    mut(inp)

## Arguments
- `inp`: The input to the mut. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the MUT. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
"""
function MUT3((input_size, hidden_size)::Pair; kwargs...)
    cell = MUT3Cell(input_size => hidden_size; kwargs...)
    return MUT3(cell)
end