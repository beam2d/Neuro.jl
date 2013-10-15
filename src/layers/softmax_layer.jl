export SoftmaxLayer, fprop, grad, bprop, weight

# Softmax layer. This is an output layer.
immutable SoftmaxLayer{T} <: ImmutableLayer{T}; end

function fprop{T}(::SoftmaxLayer{T}, in::Array{T})
    out = exp(in .- mapslices(max, in, 1))
    out ./ mapslices(sum, out, 1)
end

bprop{T}(::SoftmaxLayer{T}, in::Array{T}, out::Array{T}, groundtruth::Array{T}) = out - groundtruth
