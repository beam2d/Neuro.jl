#!/usr/bin/env julia
include("src/Neuro.jl")

function gather_source_files(dir::String, files::Array{String})
    try
        for fn in readdir(dir)
            path = joinpath(dir, fn)
            if endswith(path, ".jl")
                push!(files, path)
            else !endswith(path, ".txt")
                gather_source_files(path, files)
            end
        end
    catch e
        # Exception is thrown when readdir failed since dir is not a directory.
    end
end

jl_list = String[]
gather_source_files("test", jl_list)

for jl in jl_list
    println("testing $(jl)")
    include(jl)
end
