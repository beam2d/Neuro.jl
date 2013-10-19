using Base.Test
using Neuro.Dataset

let
    cases = [
        (Float64, "libsvm_test1.txt", 0,
         [1. 2. 3.;
          1. 2. 3.;
          1. 2. 3.],
         [1, 2, 3]),
        (Float64, "libsvm_test1.txt", 4,
         [1. 2. 3.;
          1. 2. 3.;
          1. 2. 3.;
          0. 0. 0.;],
         [1, 2, 3]),
        (Float64, "libsvm_test2.txt", 0,
         [1. 0. 2.;
          0. 3. 1.;
          3. 1. 0.],
         [1, 3, 2]),
        (Float32, "libsvm_test1.txt", 0,
         [1.f0 2.f0 3.f0;
          1.f0 2.f0 3.f0;
          1.f0 2.f0 3.f0],
         [1, 2, 3])
    ]

    for (t, filename, dim, X_expect, y_expect) in cases
        fn = "test/dataset/test_data/" * filename
        X, y = if dim == 0
            read_libsvm_dataset(t, fn)
        else
            read_libsvm_dataset(t, fn, dim)
        end

        @test X_expect == X
        @test y_expect == y
    end
end
