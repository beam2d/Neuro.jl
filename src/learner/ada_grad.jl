export AdaGrad, update!

# AdaGrad algorithm
# J. Duchi, E. Hazan and Y. Singer. Adaptive Subgradient Methods for Online
# Learning and Stochastic Optimization. COLT, 2010.

immutable AdaGrad{T<:Real} <: AbstractLearner{T}
    weights::Array{Array{T}}
    sqsum_grads::Array{Array{T}}
    rate::T

    # eps is the initial value of elements of sqsum_grads. Using small but
    # positive eps gives arithmetic stability at the first few updates.
    AdaGrad{T}(weights::Array{Array{T}}, rate::T, eps::T=convert(T, 0.01)) =
        new(weights, Array{T}[fill(eps, size(w)) for w in weights], rate)
end

function update!{T}(ada_grad::AdaGrad{T}, grads::Array{Array{T}})
    @assert length(ada_grad.sqsum_grads) == length(grads)
    for w, ag, g in zip(ada_grad.weights, ada_grad.sqsum_grads, grads)
        ag += abs2(g)
        w -= ada_grad.rate * g ./ sqrt(ag)
    end
end
