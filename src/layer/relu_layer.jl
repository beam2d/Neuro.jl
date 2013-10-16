export ReluLayer, fprop, grad, bprop, weight

# Rectified linear layer.
immutable ReluLayer{T} <: ImmutableLayer{T}; end

fprop{T}(::ReluLayer{T}, in::Array{T}) = max(in, 0)
bprop{T}(::ReluLayer{T}, in::Array{T}, out::Array{T}, d_out::Array{T}) = (in .> 0) .* d_out
