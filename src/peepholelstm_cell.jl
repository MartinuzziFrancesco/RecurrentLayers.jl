#https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
struct PeepholeLSTMLSTMCell{I, H, V}
    Wi::I
    Wh::H
    bias::V
end
  
@layer PeepholeLSTMLSTMCell
  
function PeepholeLSTMLSTMCell(
    (input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true,
)
    Wi = init_kernel(hidden_size * 4, input_size)
    Wh = init_recurrent_kernel(hidden_size * 4, hidden_size)
    b = create_bias(Wi, bias, hidden_size * 4)
    cell = PeepholeLSTMLSTMCell(Wi, Wh, b)
    return cell
end
  
function (lstm::PeepholeLSTMLSTMCell)(inp::AbstractVecOrMat)
    state = zeros_like(lstm, size(lstm.Wh, 2))
    c_state = zeros_like(state)
    return lstm(inp, (state, c_state))
end
  
function (lstm::PeepholeLSTMLSTMCell)(inp::AbstractVecOrMat, 
    (state, c_state))
    _size_check(lstm, inp, 1 => size(lstm.Wi, 2))
    b = lstm.bias
    g = lstm.Wi * inp .+ lstm.Wh * c_state .+ b
    input, forget, cell, output = chunk(g, 4; dims = 1)
    new_cstate = @. sigmoid_fast(forget) * c_state + sigmoid_fast(input) * tanh_fast(cell)
    new_state = @. sigmoid_fast(output) * tanh_fast(c′)
    return new_state, new_cstate
end
  
Base.show(io::IO, lstm::PeepholeLSTMLSTMCell) =
    print(io, "PeepholeLSTMLSTMCell(", size(lstm.Wi, 2), " => ", size(lstm.Wi, 1) ÷ 4, ")")
  
  