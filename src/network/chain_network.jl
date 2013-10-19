export ChainNetwork, fprop, calc_activations, grad, weight

import ..Layer

# Chain-shaped neural network, including usual deep classifiers.
typealias ChainNetwork{T} Vector{Layer.AbstractLayer{T}}

function fprop{T}(chain::ChainNetwork{T}, in::AbstractArray{T})
    for layer in chain
        in = Layer.fprop(layer, in)
    end
    in
end

function calc_activations{T}(chain::ChainNetwork{T}, in::AbstractArray{T})
    activations = Array{T}[in]
    for layer in chain
        push!(activations, Layer.fprop(layer, activations[end]))
    end
    activations
end

function grad{T}(chain::ChainNetwork{T}, activations::Vector{Array{T}}, d::AbstractArray{T})
    grads = Layer.AbstractLayer[]
    for j=length(chain):-1:1
        push!(grads, Layer.grad(chain[j], activations[j], d))
        d = Layer.bprop(chain[j], activations[j], activations[j + 1], d)
    end
    reverse(grads)
end

function weight{T}(chain::ChainNetwork{T})
    weights = Array{T}[]
    for layer in chain
        append!(weights, Layer.weight(layer))
    end
    weights
end
