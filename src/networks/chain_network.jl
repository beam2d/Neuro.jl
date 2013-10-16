export ChainNetwork, fprop, calc_activations, grad, weight

import ..Layers

# Chain-shaped neural network, including usual deep classifiers.
typealias ChainNetwork{T} Vector{Layer{T}}

function fprop{T}(chain::ChainNetwork{T}, in::Array{T})
    for layer in chain
        in = Layers.fprop(layer, in)
    end
    in
end

function calc_activations{T}(chain::ChainNetwork{T}, in::Array{T})
    activations = Array{T}[in]
    for layer in chain
        push!(activations, Layers.fprop(layer, activations[end]))
    end
    activations
end

function grad{T}(chain::ChainNetwork{T}, activations::Vector{Array{T}}, d::Array{T})
    grads = Layer[]
    for j=length(chain):-1:1
        push!(grads, Layers.grad(chain[j], activations[j], d))
        d = Layers.bprop(chain[j], activations[j], activations[j + 1], d)
    end
    reverse(grads)
end

function weight{T}(chain::ChainNetwork{T})
    weights = Array{T}[]
    for layer in chain
        append!(weights, Layers.weight(layer))
    end
    weights
end
