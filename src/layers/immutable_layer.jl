export ImmutableLayer, grad, weight

# Base abstract type for non-learning (immutable) layers.
abstract ImmutableLayer{T} <: Layer{T}

grad{T}(l::ImmutableLayer{T}, in::Array{T}, d_out::Array{T}) = l
weight{T}(::ImmutableLayer{T}) = Array{T}[]
