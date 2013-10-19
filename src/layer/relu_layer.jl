export ReluLayer, fprop, grad, bprop, weight

# Rectified linear layer.
immutable ReluLayer{T} <: ImmutableLayer{T}; end

fprop{T}(::ReluLayer{T}, in::AbstractArray{T}) = max(0, in)
bprop{T}(::ReluLayer{T}, in::AbstractArray{T}, out::AbstractArray{T}, d_out::AbstractArray{T}) =
    (in .> 0) .* d_out
