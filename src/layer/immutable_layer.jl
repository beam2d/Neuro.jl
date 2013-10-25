export ImmutableLayer, grad, weight

# Base abstract type for non-learning (immutable) layers.
abstract ImmutableLayer{T} <: AbstractLayer{T}

grad{T}(l::ImmutableLayer{T}, in::Array{T}, d_out::Array{T}) = Array{T}[]
weight{T}(::ImmutableLayer{T}) = Array{T}[]
