using Base.Test
using Neuro.Layers

@test_throws DenseLayer([1. 2.; 3. 4.], [1. 2. 3.])
@test_throws DenseLayer([1. 2. 3.; 4. 5. 6.], [1. 2.])
@test_throws DenseLayer(zeros(Float64, (0, 0)), Float64[])

# Test in/out size
let
    l = DenseLayer(randn(2, 3), randn(2))
    @test 3 == in_size(l)
    @test 2 == out_size(l)
end

# Usual test case
let
    W = [1. 0. -1.; 0. 1. -1.]
    b = [1., -1.]
    l = DenseLayer(W, b)

    # Test on a sample
    let
        x = [1., 2., 3.]
        y = fprop(l, x)
        @test_approx_eq(W * x + b, y)

        d = [1., -1.]
        g = grad(l, x, d)
        @test_approx_eq(d * x', g.weight)
        @test_approx_eq(d, g.bias)

        @test_approx_eq(W' * d, bprop(l, x, y, d))
    end

    # Test on a minibatch
    let
        x1 = [1., 2., 3.]
        x2 = [-1., 0., 2.]
        y = fprop(l, [x1 x2])
        @test_approx_eq([(W * x1 + b) (W * x2 + b)], y)

        d1 = [1., -1.]
        d2 = [-2., 3.]
        g = grad(l, [x1 x2], [d1 d2])
        @test_approx_eq(d1 * x1' + d2 * x2', g.weight)
        @test_approx_eq(d1 + d2, g.bias)

        @test_approx_eq([(W' * d1) (W' * d2)], bprop(l, [x1 x2], y, [d1 d2]))
    end
end

# Weight update
let
    W = [1. 2. 3.; 4. 5. 6.]
    b = [7., 8.]
    l = DenseLayer(W, b)
    @test W === l.weight
    @test b === l.bias

    W2, b2 = weight(l)
    @test W2 === l.weight
    @test b2 === l.bias
end
