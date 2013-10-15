export ReluLayer, fprop, grad, bprop, weight

# Rectified linear layer.
immutable ReluLayer{T} <: Layer{T}; end

fprop{T}(::ReluLayer{T}, in::Array{T}) = max(in, 0)
grad{T}(l::ReluLayer{T}, in::Array{T}, d_out::Array{T}) = l
bprop{T}(::ReluLayer{T}, in::Array{T}, out::Array{T}, d_out::Array{T}) = (in .> 0) .* d_out
weight{T}(::ReluLayer{T}) = Array{T}[]
