export SoftmaxLayer, fprop, grad, bprop, weight

# Softmax layer. This is an output layer.
immutable SoftmaxLayer{T} <: Layer{T}; end

function fprop{T}(::SoftmaxLayer{T}, in::Array{T})
    out = exp(in .- mapslices(max, in, 1))
    out ./ mapslices(sum, out, 1)
end

grad{T}(l::SoftmaxLayer{T}, in::Array{T}, groundtruth::Array{T}) = l
bprop{T}(::SoftmaxLayer{T}, in::Array{T}, out::Array{T}, groundtruth::Array{T}) = out - groundtruth
weight{T}(::SoftmaxLayer{T}) = Array{T}[]
