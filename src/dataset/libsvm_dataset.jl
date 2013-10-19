# Utilities to read libsvm datasets.
export
    read_libsvm_line, update_by_libsvm_line!,
    get_libsvm_dimension, get_num_lines,
    read_libsvm_dataset

function read_libsvm_line{T}(::Type{T}, line::String)
    cols = split(strip(line))
    elems = Array((Int, T), 0)
    for col in cols[2:]
        index, value = split(col, ':')
        push!(elems, (int(index), float64(value)))
    end
    int(cols[1]), elems
end

function update_by_libsvm_line!{T}(v::AbstractArray{T}, a::AbstractArray{(Int, T)})
    fill!(v, zero(T))
    for (index, value) in a
        v[index] = value
    end
    v
end

function get_libsvm_dimension(filename::String)
    dim = 0
    open(filename) do f
        for line in eachline(f)
            label, elems = read_libsvm_line(Float64, line)
            if dim < elems[end][1]
                dim = elems[end][1]
            end
        end
    end
    dim
end

function get_num_lines(filename::String)
    count = 0
    open(filename) do f
        for line in eachline(f)
            if length(line) > 0
                count += 1
            end
        end
    end
    count
end

function read_libsvm_dataset{T}(::Type{T}, filename::String, dim::Int)
    num_samples = get_num_lines(filename)

    X = Array(T, (dim, num_samples))
    y = Array(Int, num_samples)

    open(filename) do f
        count = 1
        for line in eachline(f)
            label, elems = read_libsvm_line(T, line)
            update_by_libsvm_line!(slice(X, :, count), elems)
            y[count] = label
            count += 1
        end
    end

    X, y
end

function read_libsvm_dataset{T}(::Type{T}, filename::String)
    read_libsvm_dataset(T, filename, get_libsvm_dimension(filename))
end
