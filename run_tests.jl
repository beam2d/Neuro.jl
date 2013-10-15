#!/usr/bin/env julia
include("src/Neuro.jl")

function gather_source_files(dir::String, files::Array{String})
    for fn in readdir(dir)
        path = joinpath(dir, fn)
        if endswith(path, ".jl")
            push!(files, path)
        else
            gather_source_files(path, files)
        end
    end
end

jl_list = String[]
gather_source_files("test", jl_list)

for jl in jl_list
    println("testing $(jl)")
    include(jl)
end
