export DenseLayer, out_size, in_size, fprop, grad, bprop, weight

# Fully-connected learnable layer.
# Note that, though this type is declared as immutable, elements of its weight
# and bias are mutable.
immutable DenseLayer{T} <: AbstractLayer{T}
    weight::Matrix{T}
    bias::Vector{T}

    function DenseLayer(weight::Matrix{T}, bias::Vector{T})
        @assert size(weight, 1) == size(bias, 1) > 0
        @assert size(weight, 2) > 0
        new(weight, bias)
    end
end

DenseLayer{T}(weight::Matrix{T}, bias::Vector{T}) = DenseLayer{T}(weight, bias)

# Randomly initializes a dense layer of given size by normal distribution.
function DenseLayer{T}(in_size::Integer, out_size::Integer, deviation::T, bias::T = zero(T))
    weight::Matrix{T} = randn(out_size, in_size) * deviation
    DenseLayer(weight, fill(bias, out_size))
end

out_size(l::DenseLayer) = size(l.weight, 1)
in_size(l::DenseLayer) = size(l.weight, 2)

# Neural net operations
fprop{T}(l::DenseLayer{T}, in::AbstractArray{T}) = l.weight * in .+ l.bias
grad{T}(l::DenseLayer{T}, in::AbstractArray{T}, d_out::AbstractArray{T}) =
    DenseLayer(d_out * in', sum(d_out, 2)[:, 1])
bprop{T}(l::DenseLayer{T}, in::AbstractArray{T}, out::Array{T}, d_out::Array{T}) = l.weight' * d_out

# Extracts weights
weight{T}(l::DenseLayer{T}) = Array{T}[l.weight, l.bias]
