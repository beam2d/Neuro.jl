export ReluLayer, fprop, grad, bprop, weight

# Rectified linear layer.
immutable ReluLayer{T} <: ImmutableLayer{T}; end

fprop{T}(::ReluLayer{T}, in::AbstractArray{T}) = max(in, 0)
bprop{T}(::ReluLayer{T}, in::AbstractArray{T}, out::AbstractArray{T}, d_out::AbstractArray{T}) =
    (in .> 0) .* d_out
